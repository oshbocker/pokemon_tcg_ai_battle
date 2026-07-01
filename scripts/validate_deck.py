"""Engine-level legality + playability check for a raw 60-card deck CSV.

Unlike check_legal (which only confirms each Card ID exists in the metadata), this
starts an ACTUAL battle with the deck on both seats — so it exercises the C
engine's deck-building rules (basic-Pokémon requirement, copy limits, etc.) — and
plays a full game with a trivial 'first legal option' policy to confirm the deck
can be piloted to a result without illegal states.

    uv run python scripts/validate_deck.py agent/decks/coevo_seeds/grimmsnarl.csv --games 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO / "agent"


def load_deck(path: Path) -> list[int]:
    ids = [int(x) for x in path.read_text().split() if x.strip()]
    if len(ids) != 60:
        raise SystemExit(f"{path}: expected 60 ids, got {len(ids)}")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", type=Path)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=4000)
    a = ap.parse_args()
    deck = load_deck(a.deck if a.deck.is_absolute() else REPO / a.deck)

    os.chdir(AGENT_DIR)
    sys.path.insert(0, str(AGENT_DIR))
    from cg.api import to_observation_class as to_oc  # type: ignore[reportMissingImports]
    from cg.game import (  # type: ignore[reportMissingImports]
        battle_finish,
        battle_select,
        battle_start,
    )

    results, decisions = [], 0
    for g in range(a.games):
        obs, result = None, None
        try:
            obs, _ = battle_start(deck, deck)
            if obs is None:
                raise SystemExit(f"ENGINE REJECTED DECK {a.deck} (battle_start returned None)")
            for _ in range(a.max_steps):
                oc = to_oc(obs)
                cur = oc.current
                if cur is None:
                    break
                if cur.result is not None and cur.result >= 0:
                    result = cur.result
                    break
                # trivial policy: take the first minCount legal option indices
                sel = list(range(oc.select.minCount)) or [0]
                decisions += 1
                obs = battle_select(sel)
        finally:
            if obs is not None:
                battle_finish()
        if result is None:
            raise SystemExit(f"game {g}: no result in {a.max_steps} steps")
        results.append(result)
    print(
        f"OK  {a.deck.name}: engine accepted deck; {a.games} games, {decisions} decisions, "
        f"results={results} (0=seat0,1=seat1,2=draw)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
