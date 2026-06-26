#!/usr/bin/env python
"""Package an agent into a competition-ready `submission.tar.gz`.

The archive must contain, at the TOP LEVEL (not nested in a folder):

    submission.tar.gz
    ├── main.py        # entry point; defines `agent(obs_dict) -> list[int]`
    ├── deck.csv       # 60 Card IDs, one per line
    └── cg/            # the simulator API package from the sample submission

`cg/` comes from the competition's sample submission (download it with
`uv run python scripts/kaggle_tool.py download-data`, then point --cg at the
`sample_submission/cg` directory, or copy it under `agent/cg`).

    uv run python scripts/build_submission.py \
        --main agent/main.py --deck agent/deck.csv --cg agent/cg

Mirrors the packaging in the official "From Deck to First Valid Submission"
notebook and the bundle-builder pattern from the orbit_wars repo.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def build(main: Path, deck: Path, cg: Path, out: Path) -> Path:
    for label, p in [("main.py", main), ("deck.csv", deck), ("cg/", cg)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found at {p}")
    lines = [ln for ln in deck.read_text().splitlines() if ln.strip()]
    if len(lines) != 60:
        raise ValueError(f"deck must have exactly 60 Card IDs, got {len(lines)}")

    def _no_pycache(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        # Strip caches/compiled artifacts — some validators reject them.
        base = Path(info.name).name
        if "__pycache__" in Path(info.name).parts or base.endswith((".pyc", ".pyo")):
            return None
        return info

    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(main, arcname="main.py")
        tar.add(deck, arcname="deck.csv")
        tar.add(cg, arcname="cg", filter=_no_pycache)

    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    print(f"created {out}")
    print("contents:")
    for n in names:
        print("  ", n)
    if "main.py" not in names:
        raise RuntimeError("main.py must be at the archive root")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default="agent/main.py", type=Path)
    ap.add_argument("--deck", default="agent/deck.csv", type=Path)
    ap.add_argument("--cg", default="agent/cg", type=Path)
    ap.add_argument("--out", default=str(REPO / "submissions" / "submission.tar.gz"), type=Path)
    args = ap.parse_args()
    build(args.main, args.deck, args.cg, args.out)


if __name__ == "__main__":
    main()
