#!/usr/bin/env python
"""One CLI to drive the Kaggle Pokémon TCG AI Battle competition.

    uv run python scripts/kaggle_tool.py leaderboard --top 20
    uv run python scripts/kaggle_tool.py submissions
    uv run python scripts/kaggle_tool.py submit submission.tar.gz -m "first agent"
    uv run python scripts/kaggle_tool.py kernels --search lucario
    uv run python scripts/kaggle_tool.py pull-kernel kiyotah/a-sample-rule-based-agent
    uv run python scripts/kaggle_tool.py topics --sort top --page-size 20   # discussions
    uv run python scripts/kaggle_tool.py topic 708586                       # read a thread
    uv run python scripts/kaggle_tool.py pages --save                       # rules -> outputs/pages
    uv run python scripts/kaggle_tool.py files
    uv run python scripts/kaggle_tool.py episodes <submission_id>
    uv run python scripts/kaggle_tool.py replay <episode_id>
    uv run python scripts/kaggle_tool.py logs <episode_id> 0

Requires ~/.kaggle/kaggle.json. Data downloads require accepting the rules on
the competition page first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptcg_battle import KaggleClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("leaderboard", help="download + print the public ladder")
    p.add_argument("--top", type=int, default=20)

    sub.add_parser("submissions", help="list our team's submissions")

    p = sub.add_parser("submit", help="submit a submission.tar.gz bundle")
    p.add_argument("bundle")
    p.add_argument("-m", "--message", required=True)

    p = sub.add_parser("kernels", help="list community notebooks/scripts (code)")
    p.add_argument("--search", default="")
    p.add_argument("--page-size", type=int, default=50)
    p.add_argument("--sort", default="hotness")

    p = sub.add_parser("pull-kernel", help="download a kernel's source")
    p.add_argument("ref")

    p = sub.add_parser("topics", help="list discussion forum topics")
    p.add_argument("--sort", default="hot", help="hot|top|new|recent|active|relevance")
    p.add_argument("--page", type=int, default=1)

    p = sub.add_parser("topic", help="read a discussion thread by id")
    p.add_argument("topic_id")
    p.add_argument("--page-size", type=int, default=50)

    p = sub.add_parser("pages", help="competition pages (rules/overview/eval)")
    p.add_argument("--save", action="store_true", help="dump to outputs/pages")
    p.add_argument("--name", help="print one page by name")

    sub.add_parser("files", help="list competition data files")
    sub.add_parser("download-data", help="download + unzip the card-metadata files")

    p = sub.add_parser("episodes", help="list episodes for one of our submissions")
    p.add_argument("submission_id")

    p = sub.add_parser("replay", help="download an episode replay JSON")
    p.add_argument("episode_id")

    p = sub.add_parser("logs", help="download agent logs for an episode")
    p.add_argument("episode_id")
    p.add_argument("agent_index", nargs="?", default="0")

    args = ap.parse_args()
    kg = KaggleClient()

    if args.cmd == "leaderboard":
        for rank, team, score in kg.leaderboard_top(args.top):
            print(f"{rank:>5}  {score:>8.1f}  {team}")
    elif args.cmd == "submissions":
        for s in kg.list_submissions():
            print(
                f"{s['date']:<20} {str(s['status']):<12} {str(s['publicScore']):>8}  {s['description']}"
            )
    elif args.cmd == "submit":
        print(kg.submit(args.bundle, args.message))
    elif args.cmd == "kernels":
        for k in kg.list_kernels(args.search, args.page_size, args.sort):
            print(f"{str(k['votes']):>4}  {k['ref']:<55} {k['title']}")
    elif args.cmd == "pull-kernel":
        print("pulled ->", kg.pull_kernel(args.ref))
    elif args.cmd == "topics":
        print(kg.topics_list(args.sort, args.page))
    elif args.cmd == "topic":
        print(kg.topic_show(args.topic_id, args.page_size))
    elif args.cmd == "pages":
        if args.save:
            print("saved ->", kg.save_pages())
        elif args.name:
            print(kg.pages().get(args.name, f"(no page named {args.name!r})"))
        else:
            for name in kg.pages():
                print(name)
    elif args.cmd == "files":
        for f in kg.list_files():
            print(f)
    elif args.cmd == "download-data":
        print("downloaded ->", kg.download_data())
    elif args.cmd == "episodes":
        print(kg.list_episodes(args.submission_id))
    elif args.cmd == "replay":
        print("saved ->", kg.download_replay(args.episode_id))
    elif args.cmd == "logs":
        print(kg.agent_logs(args.episode_id, int(args.agent_index)))


if __name__ == "__main__":
    main()
