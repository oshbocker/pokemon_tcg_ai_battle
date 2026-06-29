"""Meta analysis from replays: classify each deck (HP-weighted main attacker),
then report archetype frequency, per-archetype win rate, and a head-to-head
matchup matrix. Decks come from steps[1][seat]['action'] (authoritative)."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "EN_Card_Data.csv", newline="") as _f:
    CARDS = {r["Card ID"]: r for r in csv.DictReader(_f)}
STAGE = "Stage (Pokémon)/Type (Energy and Trainer)"


def hp(cid: str) -> int:
    v = CARDS.get(cid, {}).get("HP", "")
    return int(v) if v and v.isdigit() else 0


def archetype(ids: list[int]) -> str:
    """Name by the main attacker: highest-HP ex/Mega Pokemon; else highest-HP
    Pokemon; with a few signature overrides for non-ex archetypes."""
    cnt = collections.Counter(str(x) for x in ids)
    if "345" in cnt:  # Crustle line -> wall/deckout
        return "Crustle"
    pokes = [k for k in cnt if "Pokémon" in CARDS.get(k, {}).get(STAGE, "")]
    exs = [
        k
        for k in pokes
        if any(t in (CARDS.get(k) or {}).get("Card Name", "") for t in ("ex", "Mega"))
    ]
    pool = exs or pokes
    if not pool:
        return "??"
    best = max(pool, key=hp)
    return CARDS[best]["Card Name"]


def decks_of(path: Path) -> tuple[list[list[int]], list | None]:
    with open(path) as f:
        R = json.load(f)
    out = []
    for seat in range(2):
        act = R["steps"][1][seat].get("action")
        out.append([int(x) for x in act] if isinstance(act, list) and len(act) == 60 else [])
    return out, R.get("rewards")


def main() -> None:
    paths = sorted((REPO / "outputs" / "replays").glob("episode-*-replay.json"))
    freq: collections.Counter = collections.Counter()
    wins: collections.Counter = collections.Counter()
    games: collections.Counter = collections.Counter()
    matchup: dict = collections.defaultdict(lambda: [0, 0])  # (A,B) -> [A_wins, total]
    n_valid = 0
    for p in paths:
        (d0, d1), rw = decks_of(p)
        if not d0 or not d1 or not rw:
            continue
        n_valid += 1
        a0, a1 = archetype(d0), archetype(d1)
        freq[a0] += 1
        freq[a1] += 1
        games[a0] += 1
        games[a1] += 1
        w = 0 if rw[0] > rw[1] else 1 if rw[1] > rw[0] else -1
        if w == 0:
            wins[a0] += 1
        elif w == 1:
            wins[a1] += 1
        if a0 != a1 and w >= 0:
            winner, loser = (a0, a1) if w == 0 else (a1, a0)
            matchup[(winner, loser)][0] += 1
            matchup[(winner, loser)][1] += 1
            matchup[(loser, winner)][1] += 1

    print(f"=== {n_valid} valid replays ===\n")
    print("ARCHETYPE         seats   games   wins   WR%")
    for a, f in freq.most_common():
        g = games[a]
        wr = 100 * wins[a] / g if g else 0
        print(f"  {a:<22} {f:>3}   {g:>4}   {wins[a]:>4}   {wr:4.0f}")

    print("\n=== matchups (winner beats loser: wins/total) ===")
    seen = set()
    for (a, b), (w, t) in sorted(matchup.items(), key=lambda kv: -kv[1][1]):
        if (b, a) in seen or t == 0:
            continue
        seen.add((a, b))
        wb = matchup[(b, a)][0]
        print(f"  {a:<20} {w:>2} - {wb:<2} {b:<20} (n={t})")


if __name__ == "__main__":
    main()
