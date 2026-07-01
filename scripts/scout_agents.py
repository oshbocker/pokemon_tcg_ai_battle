"""Scout our own live agents: pull recent ladder replays for our submissions and
report per-agent strengths / weaknesses.

For each of our COMPLETE submissions (the latest N, or explicit ids via --sub),
this downloads the R most-recent ranked episodes, figures out which seat is *ours*
(the 60-card deck fingerprint that is constant across all of a submission's games),
and prints:

  * overall decisive win-rate + record,
  * per-opponent-archetype W-L-D and WR,
  * loss "shape": game length (steps) and prize margin (how many prizes we still
    needed at the loss) — separating blowouts from close games and flagging deckouts,
  * the full loss list with opponent + length.

Replays cache under ``outputs/replays/scout/<submission_id>/`` (resumable — an
episode already on disk is not re-fetched). Archetype naming is shared with
``scripts/analyze_meta_replays.py``.

Usage::

    uv run python scripts/scout_agents.py                 # latest 2 COMPLETE subs, 35 replays each
    uv run python scripts/scout_agents.py --n 3 --replays 50
    uv run python scripts/scout_agents.py --sub 54219624 --sub 54235407
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptcg_battle.kaggle_client import KaggleClient  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
with open(REPO / "data" / "EN_Card_Data.csv", newline="") as _f:
    CARDS = {r["Card ID"]: r for r in csv.DictReader(_f)}
STAGE = "Stage (Pokémon)/Type (Energy and Trainer)"
ALAKAZAM_LINE = {741, 742, 743}  # Abra/Kadabra/Alakazam — one physical combo, named 3 ways


def _hp(cid: str) -> int:
    v = CARDS.get(cid, {}).get("HP", "")
    return int(v) if v and v.isdigit() else 0


def archetype(ids: list[int]) -> str:
    """Name a 60-card list by its main attacker (shared with analyze_meta_replays)."""
    cnt = collections.Counter(str(x) for x in ids)
    if "345" in cnt:  # Crustle line -> wall/deckout
        return "Crustle"
    if ALAKAZAM_LINE & set(ids):
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
    return CARDS[max(pool, key=_hp)]["Card Name"]


def _decks_and_result(R: dict) -> tuple[list[list[int]], list | None, int, dict | None]:
    """(both 60-card decks, rewards, n_steps, final game-state 'current')."""
    decks = []
    for seat in range(2):
        act = R["steps"][1][seat].get("action")
        decks.append([int(x) for x in act] if isinstance(act, list) and len(act) == 60 else [])
    final = None
    for st in reversed(R["steps"]):
        cur = st[0]["observation"].get("current")
        if cur and "players" in cur:
            final = cur
            break
    return decks, R.get("rewards"), len(R["steps"]), final


def _episode_ids(kg: KaggleClient, sub_id: str, limit: int) -> list[str]:
    text = kg.list_episodes(sub_id)
    ids = []
    for line in text.splitlines():
        tok = line.split()
        if tok and tok[0].isdigit():
            ids.append(tok[0])
    return ids[:limit]


def _fetch(kg: KaggleClient, sub_id: str, n_replays: int) -> Path:
    dest = REPO / "outputs" / "replays" / "scout" / str(sub_id)
    dest.mkdir(parents=True, exist_ok=True)
    eps = _episode_ids(kg, sub_id, n_replays)
    have = {p.name for p in dest.glob("*.json")}
    fetched = 0
    for ep in eps:
        if any(ep in h for h in have):
            continue
        try:
            kg.download_replay(ep, dest=dest)
            fetched += 1
        except Exception as e:  # noqa: BLE001 - best-effort, report and continue
            print(f"    ! failed {ep}: {e}")
    print(
        f"  {sub_id}: {len(eps)} recent episodes, {fetched} newly downloaded "
        f"({len(list(dest.glob('*.json')))} cached)"
    )
    return dest


def analyze(folder: Path, label: str) -> None:
    paths = sorted(folder.glob("*.json"))
    games = []
    fp_counts: collections.Counter = collections.Counter()
    for p in paths:
        with open(p) as f:
            R = json.load(f)
        decks, rw, nsteps, final = _decks_and_result(R)
        if not decks[0] or not decks[1]:
            continue
        fps = (tuple(sorted(decks[0])), tuple(sorted(decks[1])))
        for fp in set(fps):  # set() so a mirror counts the deck once per game
            fp_counts[fp] += 1
        games.append((p.name, decks, fps, rw, nsteps, final))

    if not games:
        print(f"\n##### {label}: no valid replays #####")
        return
    our_fp = fp_counts.most_common(1)[0][0]
    our_arch = archetype(list(our_fp))
    n_our = sum(1 for _name, _decks, fps, _rw, _ns, _fin in games if our_fp in fps)

    print(
        f"\n##### {label} — our deck: {our_arch}  (matched in {n_our}/{len(games)} replays) #####"
    )

    wins = losses = draws = errors = mirrors = 0
    by_opp: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    len_w: list[int] = []
    len_l: list[int] = []
    loss_margins: list[int] = []
    deckouts = 0
    loss_rows = []
    for _name, decks, fps, rw, nsteps, final in games:
        if our_fp not in fps:
            continue
        if fps[0] == our_fp and fps[1] == our_fp:
            us, mirror = 0, True
            mirrors += 1
        else:
            us, mirror = (0 if fps[0] == our_fp else 1), False
        opp = us ^ 1
        opp_name = archetype(decks[opp]) + (" (MIRROR)" if mirror else "")
        if not rw or rw[0] is None or rw[1] is None:
            errors += 1
            continue
        if rw[us] > rw[opp]:
            wins += 1
            by_opp[opp_name][0] += 1
            len_w.append(nsteps)
        elif rw[us] < rw[opp]:
            losses += 1
            by_opp[opp_name][1] += 1
            len_l.append(nsteps)
            margin = deck_ct = None
            if final:
                margin = len(final["players"][us]["prize"])  # prizes we still needed
                deck_ct = [final["players"][i]["deckCount"] for i in range(2)]
                loss_margins.append(margin)
                if 0 in deck_ct:
                    deckouts += 1
            loss_rows.append((opp_name, nsteps, margin))
        else:
            draws += 1
            by_opp[opp_name][2] += 1

    dec = wins + losses
    wr = 100 * wins / dec if dec else 0
    print(
        f"  Record: {wins}W-{losses}L-{draws}D   WR={wr:.0f}% (decisive n={dec})"
        f"   [errors:{errors} mirrors:{mirrors}]"
    )
    if len_w or len_l:
        aw = sum(len_w) / len(len_w) if len_w else 0
        al = sum(len_l) / len(len_l) if len_l else 0
        print(f"  Game length (steps): wins avg {aw:.0f}  losses avg {al:.0f}")
    if loss_margins:
        dist = dict(sorted(collections.Counter(loss_margins).items()))
        close = sum(1 for m in loss_margins if m == 1)
        blow = sum(1 for m in loss_margins if m >= 3)
        print(
            f"  Loss margin (prizes still needed): {dist}"
            f"   close(1):{close}  blowout(>=3):{blow}  deckouts:{deckouts}"
        )

    print("  -- by opponent archetype (W-L-D) --")
    for opp, (w, lo, d) in sorted(by_opp.items(), key=lambda kv: -sum(kv[1])):
        tot = w + lo + d
        owr = 100 * w / (w + lo) if (w + lo) else 0
        print(f"    {opp:<28} {w}-{lo}-{d}   n={tot}  WR={owr:.0f}%")

    if loss_rows:
        print("  -- losses --")
        for opp, nsteps, margin in loss_rows:
            mtxt = f"prizes_left={margin}" if margin is not None else ""
            print(f"    L  {opp:<26} steps={nsteps:<4} {mtxt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n", type=int, default=2, help="how many of our latest COMPLETE subs to scout"
    )
    ap.add_argument("--replays", type=int, default=35, help="recent episodes per submission")
    ap.add_argument(
        "--sub", action="append", default=[], help="explicit submission id (repeatable)"
    )
    ap.add_argument("--no-fetch", action="store_true", help="analyze cached replays only")
    args = ap.parse_args()

    kg = KaggleClient()
    if args.sub:
        subs = [(s, "") for s in args.sub]
    else:
        complete = [s for s in kg.list_submissions() if "COMPLETE" in str(s.get("status", ""))][
            : args.n
        ]
        subs = [
            (str(s["ref"]), f"{s.get('publicScore')}  {s.get('description', '')[:55]}")
            for s in complete
        ]

    print(f"Scouting {len(subs)} submission(s), {args.replays} replays each\n")
    for sub_id, desc in subs:
        folder = REPO / "outputs" / "replays" / "scout" / str(sub_id)
        if not args.no_fetch:
            folder = _fetch(kg, sub_id, args.replays)
        label = f"sub {sub_id}" + (f"  (LB {desc})" if desc else "")
        analyze(folder, label)


if __name__ == "__main__":
    main()
