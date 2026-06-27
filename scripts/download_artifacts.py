"""Download Colab training artifacts (checkpoints, eval CSVs, run logs) from Google
Drive via rclone.

Same solution as the Orbit Wars repo (`scripts/download_checkpoint.py`): Colab runs
write straight to a `MyDrive/ptcg_outputs/` tree (see `notebooks/colab_selfplay.ipynb`),
and we pull what we need back down here with rclone. Nothing large goes through git —
the env binary already lives in the repo; weights/logs live on Drive.

Requires rclone configured with a `gdrive` remote pointing at your Google Drive.
One-time setup (also in CLAUDE.md):

    sudo apt install rclone            # or: brew install rclone / curl rclone.org/install.sh | sudo bash
    rclone config                      # New remote -> name "gdrive" -> type "drive" -> OAuth

Usage:
    # List the run directories sitting on Drive
    uv run python scripts/download_artifacts.py --list

    # Pull the run logs (colab_*.txt) back into rl_research/ (the old git target)
    uv run python scripts/download_artifacts.py --logs

    # Pull a checkpoint run dir (best.pt/last.pt) into outputs/<run>/
    uv run python scripts/download_artifacts.py --run checkpoints_sp_small

    # Pull the option-rank ablation outputs (per-arm .pt + eval CSVs)
    uv run python scripts/download_artifacts.py --run ablation_sp

    # Pull the entire ptcg_outputs tree into outputs/
    uv run python scripts/download_artifacts.py --all

    # Use a different rclone remote name
    uv run python scripts/download_artifacts.py --run ablation_sp --remote mygdrive
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The Drive subtree the Colab notebook writes to (MyDrive/ptcg_outputs/...).
DRIVE_BASE = "ptcg_outputs"
DEFAULT_REMOTE = "gdrive"
DEFAULT_RUN = "checkpoints_sp_small"

# Where each kind of artifact lands locally.
LOCAL_OUTPUTS = REPO / "outputs"  # gitignored — checkpoints, eval CSVs
LOCAL_LOGS = REPO / "rl_research"  # the colab_*.txt run logs (committed)


def _check_rclone(remote: str) -> None:
    if shutil.which("rclone") is None:
        print("Error: rclone is not installed.\n")
        print("Install it:")
        print("  sudo apt install rclone        # Debian/Ubuntu")
        print("  brew install rclone             # macOS")
        print("  curl https://rclone.org/install.sh | sudo bash\n")
        print("Then configure Google Drive:")
        print("  rclone config")
        print('  -> New remote -> name: "gdrive" -> type: "drive" -> follow the OAuth prompts')
        sys.exit(1)
    result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
    remotes = [r.rstrip(":") for r in result.stdout.strip().splitlines()]
    if remote not in remotes:
        print(f"Error: rclone remote '{remote}' not found. Available: {remotes or '(none)'}")
        print("Configure it with: rclone config")
        sys.exit(1)


def _copy(src: str, dst: Path, *, flat: bool = False) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", "--progress", src, str(dst)]
    if flat:  # pull only files in this dir, not nested run subdirs
        cmd[2:2] = ["--max-depth", "1"]
    print(f"rclone copy {src} -> {dst}/")
    return subprocess.run(cmd).returncode


def list_runs(remote: str) -> None:
    print(f"Listing {remote}:{DRIVE_BASE}/")
    subprocess.run(["rclone", "lsf", "--dirs-only", f"{remote}:{DRIVE_BASE}/"])


def download_run(remote: str, run: str) -> None:
    rc = _copy(f"{remote}:{DRIVE_BASE}/{run}/", LOCAL_OUTPUTS / run)
    if rc != 0:
        print(f"Error: rclone copy failed (exit {rc}). Try --list to see available runs.")
        sys.exit(1)
    files = sorted(p.name for p in (LOCAL_OUTPUTS / run).glob("*") if p.is_file())
    print(f"Done. {len(files)} file(s) in {LOCAL_OUTPUTS / run}/")


def download_logs(remote: str) -> None:
    """Pull the colab_*.txt run logs into rl_research/ — the place the old git-push
    cell deposited them, so they can be committed as the dated experiment record."""
    rc = _copy(f"{remote}:{DRIVE_BASE}/logs/", LOCAL_LOGS, flat=True)
    if rc != 0:
        print(f"Error: rclone copy failed (exit {rc}).")
        sys.exit(1)
    logs = sorted(p.name for p in LOCAL_LOGS.glob("colab_*.txt"))
    print(f"Done. logs in {LOCAL_LOGS}/: {logs or '(none yet)'}")


def download_all(remote: str) -> None:
    rc = _copy(f"{remote}:{DRIVE_BASE}/", LOCAL_OUTPUTS)
    if rc != 0:
        print(f"Error: rclone copy failed (exit {rc}).")
        sys.exit(1)
    print(f"Done -> {LOCAL_OUTPUTS}/")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Colab artifacts from Google Drive via rclone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--run", default=DEFAULT_RUN, help=f"run dir under {DRIVE_BASE}/ (default: {DEFAULT_RUN})"
    )
    ap.add_argument("--logs", action="store_true", help="pull colab_*.txt logs into rl_research/")
    ap.add_argument(
        "--all", action="store_true", help=f"pull the whole {DRIVE_BASE}/ tree into outputs/"
    )
    ap.add_argument("--list", action="store_true", help="list run dirs on Drive")
    ap.add_argument(
        "--remote", default=DEFAULT_REMOTE, help=f"rclone remote (default: {DEFAULT_REMOTE})"
    )
    a = ap.parse_args()

    _check_rclone(a.remote)
    if a.list:
        list_runs(a.remote)
    elif a.logs:
        download_logs(a.remote)
    elif a.all:
        download_all(a.remote)
    else:
        download_run(a.remote, a.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
