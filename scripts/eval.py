#!/usr/bin/env python
"""Single `eval` command: high-n, side-swapped, unpaired evaluation.

Runs `champion` vs a fixed honest opponent suite, streaming to a resumable CSV
and reporting win rates with Wilson 95% intervals. The engine is unseedable
(P0.4), so this is the variance-by-volume + seat-side-swap discipline from
`rl_research/PHASE0_THROUGHPUT.md` and Lesson 6 — *never* act on a small-sample
win. See `src/ptcg_battle/eval_harness.py` for the mechanics.

    # Baseline + A/A null (champion=heuristic vs an identical policy = the null):
    uv run python scripts/eval.py --champion heuristic \
        --opponents mirror,random --games 400 --out outputs/eval/baseline.csv

    # Later, evaluate a trained policy against the honest suite:
    uv run python scripts/eval.py --champion model:checkpoints/best.pt \
        --opponents heuristic,random --games 3000 --out outputs/eval/run42.csv

Re-running the same command tops up to `--games` (resumable). `mirror` means "an
identical copy of the champion"; with `--champion heuristic` that row is the A/A
null whose Wilson width sets the 'don't act below this n' floor (`--floor`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptcg_battle.eval_harness import (  # noqa: E402
    NAMED_AGENTS,
    dont_act_floor,
    evaluate,
    games_for_edge,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--champion", default="heuristic", help="agent spec (heuristic|random|first|model:<path>)"
    )
    ap.add_argument(
        "--deck",
        default=None,
        help="champion deck CSV for model:/random/first (default agent/deck.csv); "
        "e.g. agent/decks/archaludon.csv. heuristic/kaggle agents pilot their own deck.",
    )
    ap.add_argument(
        "--opponents",
        default="mirror,random",
        help="comma list (mirror=copy of champion; kaggle:<name> pilots its own deck)",
    )
    ap.add_argument(
        "--games", type=int, default=400, help="total games per opponent (split 50/50 across seats)"
    )
    ap.add_argument("--workers", type=int, default=0, help="processes (0 = cpu_count-1)")
    ap.add_argument(
        "--chunk", type=int, default=25, help="games per worker task (CSV row granularity)"
    )
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/eval.csv"))
    ap.add_argument(
        "--floor", type=float, default=5.0, help="edge (pp) for the don't-act floor report"
    )
    a = ap.parse_args()

    opponents = [o.strip() for o in a.opponents.split(",") if o.strip()]
    for o in opponents:
        if o not in NAMED_AGENTS and not o.startswith(("model:", "kaggle:")):
            ap.error(
                f"unknown opponent {o!r} (allowed: {NAMED_AGENTS}, model:<path>, kaggle:<name>)"
            )

    print(
        f"eval  champion={a.champion}  opponents={opponents}  games/opp={a.games} "
        f"(side-swapped {a.games // 2}/{a.games // 2})  workers={a.workers or 'auto'}\n"
        f"planning: {games_for_edge(5.0)} games/arm for a 5pp edge, "
        f"{games_for_edge(3.0)} for 3pp (80% power)\n"
    )
    t0 = time.time()
    summaries = evaluate(
        champion=a.champion,
        opponents=opponents,
        games=a.games,
        out_csv=a.out,
        workers=a.workers,
        max_steps=a.max_steps,
        chunk=a.chunk,
        champion_deck=a.deck,
    )
    wall = time.time() - t0

    hdr = f"{'opponent':>12} {'n(dec)':>8} {'winrate':>8} {'95% CI':>16} {'draws':>6} {'err':>5} {'seats(0/1)':>11}"
    print(hdr)
    print("-" * len(hdr))
    for opp in opponents:
        s = summaries[opp]
        p, lo, hi = s.winrate_ci()
        print(
            f"{opp:>12} {s.decisive:>8} {p * 100:>7.1f}% "
            f"[{lo * 100:>5.1f},{hi * 100:>5.1f}] {s.draws:>6} {s.errors:>5} "
            f"{s.seat0_games:>5}/{s.seat1_games:<5}"
        )

    # A/A null: prefer the mirror row, else the champion-vs-self row if present.
    aa = summaries.get("mirror")
    if aa is not None and aa.decisive > 0:
        f = dont_act_floor(aa, edge_pp=a.floor)
        print(
            f"\nA/A null (champion vs identical {a.champion}):"
            f"\n  win rate {f['aa_winrate'] * 100:.1f}%  (95% CI half-width "
            f"±{f['aa_halfwidth_pp']:.1f}pp over {f['aa_decisive_games']} decisive games)"
            f"\n  don't-act floor: ~{f['floor_games_per_arm_5pp']} games/arm to call a 5pp edge, "
            f"~{f['floor_games_per_arm_3pp']} for 3pp"
        )
        if abs(f["aa_winrate"] - 0.5) * 100 > f["aa_halfwidth_pp"]:
            print(
                "  NOTE: A/A win rate is >1 CI from 50% — likely an uncorrected "
                "seat bias or too few games; widen n before trusting edges."
            )

    print(f"\n{summaries[opponents[0]].games and ''}elapsed {wall:.1f}s  ->  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
