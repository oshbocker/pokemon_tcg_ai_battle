#!/usr/bin/env python
"""Phase-3 DEFINITIVE option-rank A/B — via SELF-PLAY on the L4 (settles the default).

The Phase-2 ablation (`ablate_option_rank.py`) trained both arms vs a fixed `random`
opponent on CPU and found ON +8pp **on-distribution**, but flagged the verdict as
provisional: training-vs-random is a confound, and the lopsided OOD head-to-head was
a non-transitivity artifact, not a feature verdict (see
`rl_research/ABLATION_OPTION_RANK.md`). This script runs the deferred settle:

  * train both arms (`use_option_rank` ON / OFF) under the **stabilized Phase-3
    recipe** — distributed self-play collection (`DistributedCollector`), KL
    early-stop, LR/entropy decay, best-checkpoint gating vs a frozen last-best — so
    neither arm collapses and the comparison isn't confounded by the P2.5 wobble;
  * judge the saved best checkpoints the way we judge everything: **high-n,
    side-swapped, Wilson-CI** eval through `scripts/eval.py`'s harness, on the honest
    suite (`random`, `first`/B1, `heuristic`) **plus a model-vs-model head-to-head**
    (ON champion vs OFF opponent — both are `model:<path>` agents now).

Self-play is the real use case, so this is the on-distribution test that matters. If
ON shows the "crutch overfits in self-play" pattern (loses the honest suite and/or
the H2H), flip the default to OFF; otherwise keep ON. The script prints a verdict
and the numbers to paste into `ABLATION_OPTION_RANK.md`.

    # On the L4 (Colab). Multi-hour for `small`; see notebooks/colab_selfplay.ipynb.
    uv run --extra rl python scripts/ablate_option_rank_selfplay.py \
        --size small --workers 12 --seeds 2 --iters 80 --games-per-iter 128 \
        --eval-n 2000 --out outputs/ablation_sp

This is the throughput-hungry one: collection is the distributed pool; only run the
real `small` sweep where the env CPUs are (the L4), not on the laptop.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from ptcg_battle.dist_collector import DistributedCollector  # noqa: E402
from ptcg_battle.eval_harness import (  # noqa: E402
    MatchupSummary,
    evaluate,
    wilson_interval,
)
from ptcg_battle.model import SIZE_BANDS, PtcgNet, param_counts  # noqa: E402
from ptcg_battle.ppo import (  # noqa: E402
    PPOConfig,
    load_deck,
    play_match,
    ppo_update,
    quick_eval,
    set_seed,
)

REPO = Path(__file__).resolve().parents[1]
HONEST_SUITE = ("random", "first", "heuristic")


def _lerp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def train_arm(use_rank: bool, deck: list[int], args, seed: int, ckpt_path: Path) -> Path:
    """Train one arm via stabilized distributed self-play; save + return its best ckpt.

    Best = the last checkpoint that beat the frozen last-best by `--gate-threshold`
    (the same gating the training script uses). If nothing ever promotes, the final
    weights are saved so the arm is still evaluable."""
    set_seed(seed)
    cfg = replace(SIZE_BANDS[args.size], use_option_rank=use_rank)
    model = PtcgNet(cfg).to(args.device)
    total, nonemb = param_counts(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(epochs=args.epochs, minibatch=args.minibatch, target_kl=args.target_kl)
    lr_final = args.lr * args.lr_decay
    ent_final = args.ent_coef * args.ent_decay

    frozen_best = copy.deepcopy(model).eval()
    best_state = copy.deepcopy(model.state_dict())
    promoted = False

    def _save(state, gate_wr, it):
        torch.save({"model": state, "cfg": cfg.__dict__, "gate_wr": gate_wr, "iter": it}, ckpt_path)

    tag = "ON " if use_rank else "OFF"
    print(
        f"  [{tag} seed={seed}] {total / 1e6:.1f}M (non-emb {nonemb / 1e6:.1f}M)  "
        f"self-play  W={args.workers}  iters={args.iters}",
        flush=True,
    )
    collector = DistributedCollector(deck, n_workers=args.workers, opponent="self")
    try:
        for it in range(1, args.iters + 1):
            frac = (it - 1) / max(1, args.iters - 1)
            for pg in opt.param_groups:
                pg["lr"] = _lerp(args.lr, lr_final, frac)
            ppo_cfg.ent_coef = _lerp(args.ent_coef, ent_final, frac)

            buf = collector.collect(
                model, args.games_per_iter, device=args.device, seed=seed * 1000 + it
            )
            m = ppo_update(model, opt, buf, ppo_cfg, device=args.device)

            if it % args.gate_every == 0 or it == args.iters:
                r = play_match(
                    model, frozen_best, deck, args.gate_games, device=args.device, seed=7
                )
                if r["winrate"] == r["winrate"] and r["winrate"] > args.gate_threshold:
                    frozen_best = copy.deepcopy(model).eval()
                    best_state = copy.deepcopy(model.state_dict())
                    promoted = True
                    _save(best_state, r["winrate"], it)
                if args.verbose:
                    sr = quick_eval(model, deck, "random", 60, device=args.device, seed=7)
                    stop = "*" if m.get("stopped_kl") else ""
                    print(
                        f"    it {it:>3} N={len(buf):>5} kl={m['approx_kl']:+.3f}{stop} "
                        f"ent={m['entropy']:.3f} gate={r['winrate'] * 100:.0f}% "
                        f"vsRand={sr['winrate'] * 100:.0f}%",
                        flush=True,
                    )
    finally:
        collector.close()

    if not promoted:  # never beat the frozen start — keep the final weights anyway
        _save(model.state_dict(), float("nan"), args.iters)
    return ckpt_path


def _pool(summaries: list[MatchupSummary]) -> tuple[int, int, int, int]:
    """Sum (wins, losses, draws, errors) across per-seed summaries for one matchup."""
    w = sum(s.wins for s in summaries)
    ln = sum(s.losses for s in summaries)
    d = sum(s.draws for s in summaries)
    e = sum(s.errors for s in summaries)
    return w, ln, d, e


def _fmt(w: int, ln: int) -> str:
    p, lo, hi = wilson_interval(w, w + ln)
    return f"{p * 100:5.1f}% [{lo * 100:4.1f},{hi * 100:4.1f}] (n={w + ln})"


def _cmp_props(w1: int, l1: int, w2: int, l2: int) -> tuple[float, float]:
    import math

    n1, n2 = w1 + l1, w2 + l2
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0
    p1, p2 = w1 / n1, w2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2), ((p1 - p2) / se if se > 0 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", default="small", choices=list(SIZE_BANDS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--games-per-iter", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-decay", type=float, default=0.33, help="final LR = lr * this")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-decay", type=float, default=0.1, help="final ent = ent * this")
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--gate-every", type=int, default=5)
    ap.add_argument("--gate-games", type=int, default=200)
    ap.add_argument("--gate-threshold", type=float, default=0.55)
    ap.add_argument(
        "--eval-n", type=int, default=2000, help="games/arm/opponent (pooled over seeds)"
    )
    ap.add_argument("--eval-workers", type=int, default=0, help="CPU procs for the honest eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "ablation_sp")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    deck = load_deck()
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"option-rank SELF-PLAY ablation  size={args.size}  seeds={args.seeds}  "
        f"iters={args.iters}  games/iter={args.games_per_iter}  W={args.workers}  "
        f"device={args.device}\nhonest eval n={args.eval_n}/arm/opp (pooled over seeds), "
        f"side-swapped, Wilson 95%\n",
        flush=True,
    )
    t0 = time.time()

    # --- train both arms for every seed, saving a checkpoint each ---
    ckpts: dict[bool, list[Path]] = {True: [], False: []}
    for seed in range(args.seeds):
        for use_rank in (True, False):
            tag = "on" if use_rank else "off"
            ts = time.time()
            path = (args.out / f"{tag}_seed{seed}.pt").resolve()
            train_arm(use_rank, deck, args, seed, path)
            ckpts[use_rank].append(path)
            print(f"  saved {path.name}  ({time.time() - ts:.0f}s)", flush=True)

    # --- high-n honest-suite eval of each arm (pooled across seeds) ---
    n_each = max(1, args.eval_n // args.seeds)
    print("\n=== high-n honest eval (pooled over seeds, side-swapped, Wilson 95%) ===", flush=True)
    suite_totals: dict[bool, dict[str, tuple[int, int]]] = {True: {}, False: {}}
    for use_rank in (True, False):
        for opp in HONEST_SUITE:
            sums = []
            for i, ckpt in enumerate(ckpts[use_rank]):
                csv = args.out / f"eval_{'on' if use_rank else 'off'}_s{i}_{opp}.csv"
                s = evaluate(
                    champion=f"model:{ckpt}",
                    opponents=[opp],
                    games=n_each,
                    out_csv=csv,
                    workers=args.eval_workers,
                )[opp]
                sums.append(s)
            w, ln, _d, _e = _pool(sums)
            suite_totals[use_rank][opp] = (w, ln)

    hdr = f"{'arm':>9}  " + "  ".join(f"{o:>22}" for o in HONEST_SUITE)
    print(hdr)
    for use_rank in (True, False):
        row = "rank ON " if use_rank else "rank OFF"
        cells = [_fmt(*suite_totals[use_rank][o]) for o in HONEST_SUITE]
        print(f"{row:>9}  " + "  ".join(f"{c:>22}" for c in cells))

    # --- head-to-head: ON champion vs OFF opponent, paired by seed, pooled ---
    print("\n=== head-to-head: ON vs OFF (paired by seed, side-swapped) ===", flush=True)
    h_w = h_l = h_d = 0
    for i, (on_ckpt, off_ckpt) in enumerate(zip(ckpts[True], ckpts[False], strict=True)):
        csv = args.out / f"h2h_s{i}.csv"
        s = evaluate(
            champion=f"model:{on_ckpt}",
            opponents=[f"model:{off_ckpt}"],
            games=n_each,
            out_csv=csv,
            workers=args.eval_workers,
        )[f"model:{off_ckpt}"]
        h_w += s.wins
        h_l += s.losses
        h_d += s.draws
    print(f"  ON {_fmt(h_w, h_l)}  vs OFF   (draws={h_d})")

    # --- verdict ---
    print("\n=== verdict ===", flush=True)
    wins_on = 0
    for opp in HONEST_SUITE:
        diff, z = _cmp_props(*suite_totals[True][opp], *suite_totals[False][opp])
        verdict = "ON>OFF" if z > 1.96 else "OFF>ON" if z < -1.96 else "~tie"
        if z > 1.96:
            wins_on += 1
        elif z < -1.96:
            wins_on -= 1
        print(f"  vs {opp:>9}: ON−OFF = {diff * 100:+5.1f}pp  z={z:+.2f}  [{verdict}]")
    _, h_lo, h_hi = wilson_interval(h_w, h_w + h_l)
    h2h = "ON>OFF" if h_lo > 0.5 else "OFF>ON" if h_hi < 0.5 else "~tie"
    print(f"  head-to-head: ON wins {h_w / max(1, h_w + h_l) * 100:.1f}% of decisive  [{h2h}]")

    if wins_on > 0 and h2h != "OFF>ON":
        rec = "KEEP use_option_rank=True (ON wins the honest suite; H2H not contradicting)."
    elif wins_on < 0 or h2h == "OFF>ON":
        rec = "FLIP default to use_option_rank=False (OFF wins on-distribution — the crutch overfits)."
    else:
        rec = "INCONCLUSIVE at this n — keep ON (provisional); widen n or iters before acting."
    print(f"\nRECOMMENDATION: {rec}")
    print(f"elapsed {time.time() - t0:.0f}s  ->  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
