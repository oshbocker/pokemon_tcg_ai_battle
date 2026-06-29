"""Pull a broad sample of top-ladder replays: leaderboard --show -> per-team top
submission -> episodes -> replay. Dedupes episodes already on disk."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from ptcg_battle.kaggle_client import KaggleClient  # noqa: E402

OUT = Path("outputs/replays")
N_TEAMS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
EP_PER_TEAM = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def ints_in(line: str) -> list[str]:
    return re.findall(r"\b\d{6,}\b", line)


def main() -> None:
    kg = KaggleClient()
    have = {p.name.split("-")[1] for p in OUT.glob("episode-*-replay.json")}
    show = kg.leaderboard_show()
    team_ids = []
    for ln in show.splitlines():
        m = re.match(r"\s*(\d{6,8})\s", ln)
        if m:
            team_ids.append(m.group(1))
    team_ids = team_ids[:N_TEAMS]
    print(f"{len(team_ids)} teams; {len(have)} replays already on disk")

    want: list[str] = []
    for t in team_ids:
        try:
            subs = kg.team_submissions(t)
        except Exception as e:
            print(f"  team {t}: {e}")
            continue
        # pick the highest publicScore submission (col 3)
        best, best_score = None, -1.0
        for ln in subs.splitlines():
            cells = ln.split()
            if len(cells) >= 3 and cells[0].isdigit():
                try:
                    sc = float(cells[-1])
                except ValueError:
                    continue
                if sc > best_score:
                    best, best_score = cells[0], sc
        if not best:
            continue
        eps = kg.list_episodes(best)
        added = 0
        for ln in eps.splitlines():
            ids = re.findall(r"\b\d{8}\b", ln)
            if ids and "COMPLETED" in ln:
                ep = ids[0]
                if ep not in have and ep not in want:
                    want.append(ep)
                    added += 1
                if added >= EP_PER_TEAM:
                    break
    print(f"new episodes to pull: {len(want)}")
    for i, ep in enumerate(want, 1):
        try:
            kg.download_replay(ep)
            print(f"  [{i}/{len(want)}] {ep}")
        except Exception as e:
            print(f"  [{i}/{len(want)}] {ep} FAILED: {e}")


if __name__ == "__main__":
    main()
