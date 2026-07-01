"""Meta analysis from replays: classify each deck (HP-weighted main attacker),
then report archetype prevalence (by seat AND by distinct decklist), per-archetype
win rate, and a head-to-head matchup matrix. Decks come from steps[1][seat]['action']
(authoritative).

Two prevalence columns: `seats` is raw appearances (inflated by repeated games — e.g.
our own cached RL-agent episodes), while `pilots` counts DISTINCT 60-card lists, which
de-biases repeats and is the truer "how many teams play this" signal. See
`rl_research/META_DISTRIBUTION_2026-06-30.md`."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "EN_Card_Data.csv", newline="") as _f:
    CARDS = {r["Card ID"]: r for r in csv.DictReader(_f)}
STAGE = "Stage (Pokémon)/Type (Energy and Trainer)"

# Abra / Kadabra / Alakazam line. Every deck carrying it is the same physical
# hand-scaling OHKO combo — the HP-weighted namer otherwise splits it three ways
# ("Fezandipiti ex" / "Alakazam" / "Dudunsparce") by whichever 1-of ex it latches
# onto (verified: 100% of those labels carry this line). Merge them into one name.
ALAKAZAM_LINE = {741, 742, 743}


def hp(cid: str) -> int:
    v = CARDS.get(cid, {}).get("HP", "")
    return int(v) if v and v.isdigit() else 0


def archetype(ids: list[int]) -> str:
    """Name by the main attacker: highest-HP ex/Mega Pokemon; else highest-HP
    Pokemon; with a few signature overrides for non-ex archetypes."""
    cnt = collections.Counter(str(x) for x in ids)
    if "345" in cnt:  # Crustle line -> wall/deckout
        return "Crustle"
    if ALAKAZAM_LINE & set(ids):  # the hand-scaling combo, however the namer sees it
        return "Alakazam combo"
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
    seats: collections.Counter = collections.Counter()  # raw appearances
    wins: collections.Counter = collections.Counter()
    decisive: collections.Counter = collections.Counter()  # WR denominator (no draws/errors)
    pilots: dict[str, set] = collections.defaultdict(set)  # archetype -> {deck fingerprints}
    matchup: dict = collections.defaultdict(lambda: [0, 0])  # (A,B) -> [A_wins, total]
    n_valid = 0
    for p in paths:
        (d0, d1), rw = decks_of(p)
        if not d0 or not d1:
            continue
        n_valid += 1
        a0, a1 = archetype(d0), archetype(d1)
        for a, d in ((a0, d0), (a1, d1)):
            seats[a] += 1
            pilots[a].add(tuple(sorted(d)))
        if not rw or rw[0] is None or rw[1] is None:
            continue  # errored/incomplete: counted for prevalence, skipped for win stats
        w = 0 if rw[0] > rw[1] else 1 if rw[1] > rw[0] else -1
        if w in (0, 1):
            decisive[a0] += 1
            decisive[a1] += 1
            wins[(a0, a1)[w]] += 1
        if a0 != a1 and w in (0, 1):
            winner, loser = (a0, a1) if w == 0 else (a1, a0)
            matchup[(winner, loser)][0] += 1
            matchup[(winner, loser)][1] += 1
            matchup[(loser, winner)][1] += 1

    tot_seats = sum(seats.values())
    tot_pilots = sum(len(v) for v in pilots.values())
    print(
        f"=== {n_valid} valid replays  ({tot_seats} seats, {tot_pilots} distinct decklists) ===\n"
    )
    print("ARCHETYPE                    seats  seat%   pilots  pilot%    WR%")
    for a, pset in sorted(pilots.items(), key=lambda kv: -len(kv[1])):
        s, pil, d = seats[a], len(pset), decisive[a]
        wr = 100 * wins[a] / d if d else 0
        print(
            f"  {a:<26} {s:>4}  {100 * s / tot_seats:4.0f}   {pil:>5}  {100 * pil / tot_pilots:5.0f}   {wr:5.0f}"
        )

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
