"""Kaggle API helper for the Pokémon TCG AI Battle Challenge (Simulation).

A thin, typed wrapper around the official `kaggle` package plus the `kaggle`
CLI. It covers everything we need to drive the competition from the terminal:

  * leaderboard      — download / view the public ladder
  * submissions      — submit a `submission.tar.gz` bundle and list our agents
  * kernels (code)   — list and pull public notebooks/scripts for the comp
  * discussions      — list and read forum topics (via the 2.2+ CLI)
  * pages            — pull the competition's overview / rules / evaluation text
  * data             — list / download the card-metadata files
  * episodes         — list a submission's episodes, download replays + agent logs

The auth pattern (read ~/.kaggle/kaggle.json, export env, authenticate) is
carried over from the orbit_wars repo (scripts/replay_pulse.py). Discussions are
not in the Python API surface, so `topics_list` / `topic_show` shell out to the
`kaggle competitions topics ...` CLI introduced in kaggle 2.2.

Requires `~/.kaggle/kaggle.json` (Account → Create New API Token).
"""

from __future__ import annotations

import contextlib
import csv
import glob
import json
import os
import re
import subprocess
import zipfile
from html import unescape
from pathlib import Path
from typing import Any

#: The Simulation competition slug. The companion Strategy/Hackathon track is
#: ``pokemon-tcg-ai-battle-challenge-strategy``.
COMPETITION = "pokemon-tcg-ai-battle"

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"


def authenticate():
    """Authenticate the Kaggle Python API from ``~/.kaggle/kaggle.json``.

    Exporting the creds to the environment first lets the same credentials be
    picked up by the ``kaggle`` CLI subprocesses used for discussions.
    """
    with open(os.path.expanduser("~/.kaggle/kaggle.json")) as f:
        creds = json.load(f)
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _strip_html(html: str | None) -> str:
    text = re.sub(r"<[^>]+>", "", html or "")
    return re.sub(r"\n{3,}", "\n\n", unescape(text)).strip()


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """First present attribute among ``names`` (SDK shifts camelCase<->snake_case)."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


class KaggleClient:
    """Convenience wrapper bound to a single competition."""

    def __init__(self, competition: str = COMPETITION):
        self.competition = competition
        self.api = authenticate()

    # -- leaderboard ---------------------------------------------------------

    def leaderboard_download(self, dest: Path | None = None) -> Path:
        """Download the full public leaderboard CSV (one row per team)."""
        dest = dest or OUT / "leaderboard"
        dest.mkdir(parents=True, exist_ok=True)
        self.api.competition_leaderboard_download(self.competition, path=str(dest))
        for z in glob.glob(str(dest / "*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        return dest

    def leaderboard_ratings(self) -> dict[str, dict[str, Any]]:
        """``team -> {rank, score}`` parsed from the downloaded leaderboard CSV."""
        dest = self.leaderboard_download()
        csvs = sorted(glob.glob(str(dest / "*.csv")))
        if not csvs:
            return {}
        out: dict[str, dict[str, Any]] = {}
        with open(csvs[-1], encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                with contextlib.suppress(ValueError, KeyError):
                    out[row["TeamName"]] = {
                        "rank": int(row.get("Rank", 0) or 0),
                        "score": float(row["Score"]),
                    }
        return out

    def leaderboard_top(self, n: int = 20) -> list[tuple[int, str, float]]:
        """Top ``n`` teams as ``(rank, team, score)``, best first."""
        rows = [(v["rank"], team, v["score"]) for team, v in self.leaderboard_ratings().items()]
        rows.sort(key=lambda r: r[0] or 10**9)
        return rows[:n]

    # -- submissions ---------------------------------------------------------

    def submit(self, bundle: str | Path, message: str) -> Any:
        """Submit a ``submission.tar.gz`` bundle (main.py at archive root)."""
        bundle = Path(bundle)
        if not bundle.exists():
            raise FileNotFoundError(bundle)
        return self.api.competition_submit(str(bundle), message, self.competition)

    def list_submissions(self) -> list[dict[str, Any]]:
        """Our team's submissions, newest first."""
        subs = self.api.competition_submissions(self.competition)
        out = []
        for s in subs or []:
            out.append(
                {
                    "ref": _attr(s, "ref"),
                    "date": str(_attr(s, "date", default="")),
                    "description": _attr(s, "description", default=""),
                    "status": str(_attr(s, "status", default="")),
                    "publicScore": _attr(s, "public_score", "publicScore"),
                    "fileName": _attr(s, "file_name", "fileName"),
                }
            )
        return out

    # -- kernels (community code) -------------------------------------------

    def list_kernels(
        self, search: str = "", page_size: int = 50, sort_by: str = "hotness"
    ) -> list[dict[str, Any]]:
        """Public notebooks/scripts attached to this competition."""
        kernels = self.api.kernels_list(
            competition=self.competition,
            search=search,
            page_size=page_size,
            sort_by=sort_by,
        )
        return [
            {
                "ref": _attr(k, "ref"),
                "title": _attr(k, "title", default=""),
                "author": _attr(k, "author", default=""),
                "votes": _attr(k, "total_votes", "totalVotes"),
                "lastRun": str(_attr(k, "last_run_time", "lastRunTime", default="")),
            }
            for k in kernels or []
        ]

    def pull_kernel(self, ref: str, dest: Path | None = None) -> Path:
        """Download a kernel's source (+ metadata) to ``outputs/kernels/<ref>``."""
        dest = dest or OUT / "kernels" / ref.replace("/", "__")
        dest.mkdir(parents=True, exist_ok=True)
        self.api.kernels_pull(ref, path=str(dest), metadata=True)
        return dest

    # -- discussions (forum topics) -----------------------------------------

    def topics_list(self, sort: str = "hot", page: int = 1) -> str:
        """List forum topics for the competition (kaggle>=2.2 CLI).

        ``sort`` is one of: hot, top, new, recent, active, relevance.
        """
        return self._cli(
            "competitions", "topics", "list", self.competition, "-s", sort, "-p", str(page)
        )

    def topic_show(self, topic_id: int | str, page_size: int = 50) -> str:
        """Read a full discussion thread (topic + comments) by id (kaggle>=2.2 CLI)."""
        return self._cli(
            "competitions",
            "topics",
            "show",
            self.competition,
            str(topic_id),
            "--page-size",
            str(page_size),
        )

    # -- competition pages (rules / overview / evaluation) ------------------

    def pages(self) -> dict[str, str]:
        """``page name -> plaintext`` for every competition page (rules, etc.)."""
        pages = self.api.competition_list_pages(self.competition)
        return {p.name: _strip_html(getattr(p, "content", "")) for p in pages or [] if p}

    def save_pages(self, dest: Path | None = None) -> Path:
        """Dump all competition pages to markdown under ``outputs/pages``."""
        dest = dest or OUT / "pages"
        dest.mkdir(parents=True, exist_ok=True)
        for name, text in self.pages().items():
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "page"
            (dest / f"{slug}.md").write_text(f"# {name}\n\n{text}\n")
        return dest

    # -- data files ---------------------------------------------------------

    def list_files(self) -> list[str]:
        """Names of the competition's data files (card metadata, sample sub)."""
        resp = self.api.competition_list_files(self.competition)
        files = getattr(resp, "files", None) or []
        return [f.name for f in files]

    def download_data(self, dest: Path | None = None) -> Path:
        """Download + unzip all competition data files (requires rule acceptance)."""
        dest = dest or REPO / "data"
        dest.mkdir(parents=True, exist_ok=True)
        self.api.competition_download_files(self.competition, path=str(dest))
        for z in glob.glob(str(dest / "*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        return dest

    # -- episodes / replays -------------------------------------------------

    def leaderboard_show(self) -> str:
        """Public ladder WITH team IDs (the CSV download omits teamId).

        Use the printed ``teamId`` with :meth:`team_submissions` to reach any
        team's active submissions, then their episodes and replays.
        """
        return self._cli("competitions", "leaderboard", self.competition, "--show")

    def team_submissions(self, team_id: int | str) -> str:
        """List another team's active (public) submissions in this sim competition.

        Gateway to other teams' replays: leaderboard ``--show`` -> teamId ->
        ``team_submissions`` -> ``list_episodes`` -> ``download_replay``.
        """
        return self._cli("competitions", "team-submissions", str(team_id))

    def list_episodes(self, submission_id: int | str) -> str:
        """List episodes (games) played by a submission (ours or any team's)."""
        return self._cli("competitions", "episodes", str(submission_id))

    def download_replay(self, episode_id: int | str, dest: Path | None = None) -> Path:
        """Download an episode replay JSON to ``outputs/replays``."""
        dest = dest or OUT / "replays"
        dest.mkdir(parents=True, exist_ok=True)
        self._cli("competitions", "replay", str(episode_id), "-p", str(dest))
        return dest

    def agent_logs(self, episode_id: int | str, agent_index: int = 0) -> str:
        """Download agent logs for an episode (useful for debugging Errors)."""
        return self._cli("competitions", "logs", str(episode_id), str(agent_index))

    # -- internals ----------------------------------------------------------

    def _cli(self, *args: str) -> str:
        """Run the `kaggle` CLI, returning stdout (creds via env from auth)."""
        proc = subprocess.run(["kaggle", *args], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"`kaggle {' '.join(args)}` failed ({proc.returncode}):\n{proc.stderr.strip()}"
            )
        return proc.stdout
