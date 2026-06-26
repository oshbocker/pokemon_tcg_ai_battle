#!/usr/bin/env python
"""The single quality gate — run before every commit / submission.

Mirrors the competition winner's `just prepare` (Lesson 10: agentic dev works
*when scaffolded*). One command, deterministic order, fail-fast summary:

    1. ruff format     — apply formatting (use --check to only verify)
    2. ruff check      — lint (E,F,I,UP,B,SIM)
    3. pyright         — type-check (basic)
    4. pytest          — encoding round-trip + eval-math units

Usage:
    uv run python scripts/prepare.py            # format-in-place, then lint/type/test
    uv run python scripts/prepare.py --check    # verify-only (CI / pre-submit)

Exit code is non-zero if any stage fails, so it composes into hooks and CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent  # venv bin/ — ruff, pyright, pytest live here


def _tool(name: str) -> list[str]:
    """Prefer the venv console script; fall back to `uv run <name>`."""
    p = BIN / name
    return [str(p)] if p.exists() else ["uv", "run", name]


def _run(label: str, cmd: list[str]) -> tuple[str, bool, float]:
    print(f"\n\033[1m== {label} ==\033[0m  ({' '.join(cmd)})")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    return (label, rc == 0, time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="verify formatting only (don't rewrite files)"
    )
    a = ap.parse_args()

    fmt = _tool("ruff") + ["format"] + (["--check"] if a.check else []) + ["."]
    stages = [
        ("ruff format", fmt),
        ("ruff check", _tool("ruff") + ["check", "."]),
        ("pyright", _tool("pyright")),
        ("pytest", _tool("pytest")),
    ]

    results = []
    for label, cmd in stages:
        results.append(_run(label, cmd))

    print("\n\033[1m== prepare summary ==\033[0m")
    ok = True
    for label, passed, dt in results:
        mark = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
        print(f"  {mark}  {label:<14} {dt:5.1f}s")
        ok = ok and passed
    print("\033[32m\nprepare: OK\033[0m" if ok else "\033[31m\nprepare: FAILED\033[0m")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
