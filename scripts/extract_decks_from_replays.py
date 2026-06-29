"""Extract both players' exact 60-card decks from replay JSONs.

The cabt engine records each agent's initial-selection action (the full 60 Card
IDs) as `steps[1][seat].action`. So every replay is an authoritative source of
both decks — no inference needed. We name the archetype by its headline ex/Mega
Pokemon and validate legality via the engine.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "EN_Card_Data.csv", newline="") as _f:
    CARDS = {r["Card ID"]: r for r in csv.DictReader(_f)}


def deck_from_replay(path: Path) -> tuple[list[list[int]], list | None]:
    with open(path) as f:
        R = json.load(f)
    out = []
    for seat in range(2):
        act = R["steps"][1][seat].get("action")
        out.append([int(x) for x in act] if isinstance(act, list) and len(act) == 60 else [])
    return out, R.get("rewards")


def archetype(ids: list[int]) -> str:
    cnt = collections.Counter(str(x) for x in ids)
    # headline = most-copied Pokemon that is an ex/Mega, else most-copied Pokemon.
    # Real type lives in the "Stage (Pokémon)/Type" column (Category is mostly n/a).
    stage_col = "Stage (Pokémon)/Type (Energy and Trainer)"
    pokes = [
        (k, c) for k, c in cnt.most_common() if "Pokémon" in CARDS.get(k, {}).get(stage_col, "")
    ]
    flashy = [
        (k, c)
        for k, c in pokes
        if any(t in (CARDS.get(k) or {}).get("Card Name", "") for t in ("ex", "Mega"))
    ]
    pick = (flashy or pokes or [(None, 0)])[0][0]
    return CARDS.get(pick, {}).get("Card Name", "??") if pick else "??"


def main() -> None:
    paths = sorted((REPO / "outputs" / "replays").glob("*.json"))
    tally: collections.Counter = collections.Counter()
    for p in paths:
        (d0, d1), rewards = deck_from_replay(p)
        a0, a1 = archetype(d0), archetype(d1)
        tally[a0] += 1
        tally[a1] += 1
        win = (
            "?"
            if not rewards
            else ("p0" if rewards[0] > rewards[1] else "p1" if rewards[1] > rewards[0] else "draw")
        )
        print(f"{p.name[:28]:28}  {a0:24} vs {a1:24}  win={win}")
    print("\n=== archetype frequency across", len(paths), "replays (2 seats each) ===")
    for name, c in tally.most_common():
        print(f"  {c:>3}  {name}")


if __name__ == "__main__":
    main()
