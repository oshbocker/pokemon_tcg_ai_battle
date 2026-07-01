"""Verify a proposed decklist against the competition card pool (data/EN_Card_Data.csv).

Feed it a decklist as lines of "<count> <card name>" on stdin (or a file arg).
Reports: each card -> legal? (# printings, which expansions), any not-found (with
close-name suggestions), and whether the counts sum to 60. Name match is
case/whitespace/punctuation-insensitive (apostrophe and accent variants normalized).
"""

from __future__ import annotations

import collections
import csv
import re
import sys
import unicodedata
from difflib import get_close_matches
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
with open(REPO / "data" / "EN_Card_Data.csv", newline="") as _f:
    ROWS = list(csv.DictReader(_f))


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.replace("’", "'").replace("`", "'").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)  # drop punctuation
    return re.sub(r"\s+", " ", s).strip()


BY_NORM: dict[str, list[dict]] = collections.defaultdict(list)
for r in ROWS:
    BY_NORM[norm(r["Card Name"])].append(r)
ALL_NORMS = list(BY_NORM)


def parse(text: str) -> list[tuple[int, str]]:
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*[xX]?\s+(.*?)\s*$", line)
        if not m:
            m = re.match(r"^(.*?)\s+[xX]?\s*(\d+)\s*$", line)  # "Name x4" / "Name 4"
            if m:
                out.append((int(m.group(2)), m.group(1)))
            continue
        out.append((int(m.group(1)), m.group(2)))
    return out


def main() -> None:
    text = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read()
    deck = parse(text)
    total = 0
    legal = missing = 0
    print(f"{'cnt':>3}  {'card':<36} status")
    print("-" * 70)
    for cnt, name in deck:
        total += cnt
        key = norm(name)
        if key in BY_NORM:
            printings = BY_NORM[key]
            exps = sorted({p["Expansion"] for p in printings})
            print(f"{cnt:>3}  {name:<36} OK  [{','.join(exps)}]")
            legal += cnt
        else:
            sugg = get_close_matches(key, ALL_NORMS, n=3, cutoff=0.7)
            sugg_names = [BY_NORM[s][0]["Card Name"] for s in sugg]
            print(f"{cnt:>3}  {name:<36} NOT FOUND  ~ {sugg_names}")
            missing += cnt
    print("-" * 70)
    print(
        f"total cards: {total}  (legal {legal} / missing {missing})   "
        f"{'OK 60' if total == 60 else '!! NOT 60'}"
    )


if __name__ == "__main__":
    main()
