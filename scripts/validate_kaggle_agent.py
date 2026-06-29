#!/usr/bin/env python
"""Validate a vendored Kaggle pool agent: it loads, carries its own legal 60-card
deck, and plays a full self-game without throwing or returning an illegal selection.

    uv run python scripts/validate_kaggle_agent.py archaludon
    uv run python scripts/validate_kaggle_agent.py dragapult --games 3

`<name>` resolves to `agent/kaggle_agents/<name>.py`. The agent plays BOTH seats of
a side-swapped self-game with its own deck; every `agent(obs)` return is checked
against the engine's option count + min/max bounds (the submission contract). This
is the torch-free gate the league relies on — a borrowed agent that crashes or
returns an illegal move would forfeit games and poison the opponent pool.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO / "agent"
KAGGLE_AGENT_DIR = AGENT_DIR / "kaggle_agents"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_deck(deck) -> list[int]:
    if not isinstance(deck, list) or len(deck) != 60:
        raise AssertionError(
            f"deck must be a list of 60 Card IDs, got {type(deck)} len={len(deck) if isinstance(deck, list) else '?'}"
        )
    if not all(isinstance(c, int) for c in deck):
        raise AssertionError("deck must be all ints")
    return deck


def _check_selection(sel, n: int, lo: int, hi: int) -> None:
    if not isinstance(sel, list):
        raise AssertionError(f"selection must be a list, got {type(sel)}")
    if any((not isinstance(i, int)) or i < 0 or i >= n for i in sel):
        raise AssertionError(f"selection {sel} has an out-of-range index (n={n})")
    if len(set(sel)) != len(sel):
        raise AssertionError(f"selection {sel} has duplicates")
    if not (lo <= len(sel) <= hi):
        raise AssertionError(f"selection len {len(sel)} not in [{lo},{hi}]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("name", help="agent module name under agent/kaggle_agents/")
    ap.add_argument("--games", type=int, default=2, help="self-games to play (side-swapped)")
    ap.add_argument("--max-steps", type=int, default=4000)
    a = ap.parse_args()

    path = KAGGLE_AGENT_DIR / f"{a.name}.py"
    if not path.exists():
        ap.error(f"no such agent: {path}")

    os.chdir(AGENT_DIR)
    sys.path.insert(0, str(AGENT_DIR))
    from cg.api import to_observation_class as to_oc  # type: ignore[reportMissingImports]
    from cg.game import (  # type: ignore[reportMissingImports]
        battle_finish,
        battle_select,
        battle_start,
    )

    mod = _load_module(path, f"validate_{a.name}")
    agent = mod.agent
    deck = _check_deck(mod.my_deck)
    # The deck-selection contract: at selection the engine emits an empty obs
    # (current=None, select=None, logs=[]); agent must return the 60-card deck.
    sel0 = agent({"logs": [], "current": None, "select": None})
    _check_deck(sel0)
    print(f"loaded {a.name}: deck OK (60 cards, first 5 {deck[:5]}), deck-selection OK")

    decisions = 0
    results: list[int] = []
    for g in range(a.games):
        obs = None
        result = None
        try:
            obs, _ = battle_start(deck, deck)
            if obs is None:
                raise AssertionError("battle_start returned None")
            for _ in range(a.max_steps):
                oc = to_oc(obs)
                cur = oc.current
                if cur is None:
                    break
                if cur.result is not None and cur.result >= 0:
                    result = cur.result
                    break
                sel = agent(obs)
                _check_selection(sel, len(oc.select.option), oc.select.minCount, oc.select.maxCount)
                decisions += 1
                obs = battle_select(sel)
        finally:
            if obs is not None:
                battle_finish()
        if result is None:
            raise AssertionError(f"game {g} produced no result in {a.max_steps} steps")
        results.append(result)
        print(f"  game {g}: result={result}  (0=seat0 win, 1=seat1 win, 2=draw)")

    print(
        f"\nOK — {a.name} played {a.games} self-games, {decisions} legal decisions, results={results}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
