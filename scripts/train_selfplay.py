#!/usr/bin/env python
"""Single-process self-play PPO training loop (Phase 2, P2.5).

The end-to-end sanity loop: collect → PPO update → periodically eval win-rate vs
the honest baselines, checkpointing the best. The goal of this phase is to confirm
the whole pipeline learns — e.g. `--opponent random` should climb to a decisive
win-rate — before Phase 3 swaps in the distributed collector and real self-play
league. See `src/ptcg_battle/ppo.py`.

    # Loop sanity: learn to beat the random agent (fast signal).
    uv run --extra rl python scripts/train_selfplay.py --opponent random \
        --iters 40 --games-per-iter 64 --eval-games 80

    # True self-play (both seats = current net), eval vs random + heuristic.
    uv run --extra rl python scripts/train_selfplay.py --opponent self \
        --iters 200 --games-per-iter 128 --size small --device cuda

Trust `scripts/eval.py` (high-n, side-swapped, Wilson CIs) for real decisions; the
in-loop eval here is a low-n gut check.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from ptcg_battle.model import SIZE_BANDS, ModelConfig, PtcgNet, param_counts  # noqa: E402
from ptcg_battle.ppo import (  # noqa: E402
    PPOConfig,
    adapt_ent_coef,
    collect_rollout,
    load_deck,
    play_match,
    ppo_update,
    quick_eval,
    set_seed,
)

REPO = Path(__file__).resolve().parents[1]


def _lerp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", default="small", choices=list(SIZE_BANDS))
    ap.add_argument("--no-option-rank", action="store_true", help="ablate the engine-order feature")
    ap.add_argument(
        "--opponent", default="random", choices=["self", "random", "first", "heuristic"]
    )
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument(
        "--epochs", type=int, default=2, help="fixed PPO passes/iter (update-size control)"
    )
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--lr-final", type=float, default=None, help="linear-decay LR target (default: lr)"
    )
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument(
        "--ent-coef", type=float, default=0.02, help="initial entropy coef (controller floor)"
    )
    ap.add_argument(
        "--ent-final", type=float, default=None, help="linear-decay target (if --target-entropy<=0)"
    )
    ap.add_argument(
        "--target-entropy",
        type=float,
        default=0.1,
        help="adaptive-entropy setpoint (mean nats); auto-tunes ent_coef. <=0 = use --ent-final decay",
    )
    ap.add_argument("--ent-gain", type=float, default=0.4, help="adaptive-entropy controller gain")
    ap.add_argument(
        "--target-kl", type=float, default=1.5, help="per-minibatch KL circuit breaker (0=off)"
    )
    # P3.1 distributed collector + P3.3 best-checkpoint gating.
    ap.add_argument("--collector", default="single", choices=["single", "dist"])
    ap.add_argument("--workers", type=int, default=8, help="dist collector worker procs")
    ap.add_argument(
        "--gate", action="store_true", help="promote best.pt by gating vs frozen last-best"
    )
    ap.add_argument("--gate-every", type=int, default=5)
    ap.add_argument("--gate-games", type=int, default=200, help="side-swapped games vs frozen best")
    ap.add_argument("--gate-threshold", type=float, default=0.55, help="win-rate to promote")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=80)
    ap.add_argument("--eval-opponents", default="random", help="comma list for in-loop eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "checkpoints")
    a = ap.parse_args()

    set_seed(a.seed)
    cfg = SIZE_BANDS[a.size]
    cfg = ModelConfig(**{**cfg.__dict__, "use_option_rank": not a.no_option_rank})
    model = PtcgNet(cfg).to(a.device)
    total, nonemb = param_counts(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ppo_cfg = PPOConfig(
        epochs=a.epochs, minibatch=a.minibatch, ent_coef=a.ent_coef, lr=a.lr, target_kl=a.target_kl
    )
    deck = load_deck()
    eval_opps = [o.strip() for o in a.eval_opponents.split(",") if o.strip()]
    a.out.mkdir(parents=True, exist_ok=True)
    lr_final = a.lr if a.lr_final is None else a.lr_final
    ent_final = a.ent_coef if a.ent_final is None else a.ent_final
    adaptive_ent = a.target_entropy > 0  # controller vs. fixed decay schedule

    # P3.1: persistent distributed worker pool (built once, reused every iter).
    collector = None
    if a.collector == "dist":
        from ptcg_battle.dist_collector import DistributedCollector

        collector = DistributedCollector(
            deck, n_workers=a.workers, opponent=a.opponent, max_steps=4000
        )

    # P3.3: frozen last-best opponent for gated promotion (the anti-collapse net).
    frozen_best = copy.deepcopy(model).eval() if a.gate else None

    print(
        f"train  size={a.size}({total / 1e6:.1f}M, non-emb {nonemb / 1e6:.1f}M)  "
        f"option_rank={cfg.use_option_rank}  opponent={a.opponent}  device={a.device}\n"
        f"       iters={a.iters} games/iter={a.games_per_iter} epochs={a.epochs} mb={a.minibatch} "
        f"lr={a.lr}->{lr_final} "
        + (
            f"ent~target={a.target_entropy}(adaptive)"
            if adaptive_ent
            else f"ent={a.ent_coef}->{ent_final}"
        )
        + f" kl_cut={a.target_kl} gamma={a.gamma}\n"
        f"       collector={a.collector}"
        + (f"(W={a.workers})" if a.collector == "dist" else "")
        + (f"  gate>{a.gate_threshold:.0%}/{a.gate_games}g every {a.gate_every}" if a.gate else "")
        + "\n"
    )
    best = -1.0
    try:
        for it in range(1, a.iters + 1):
            frac = (it - 1) / max(1, a.iters - 1)
            cur_lr = _lerp(a.lr, lr_final, frac)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            if not adaptive_ent:  # fixed decay; adaptive coef is updated after the step
                ppo_cfg.ent_coef = _lerp(a.ent_coef, ent_final, frac)

            t0 = time.time()
            if collector is not None:
                buf = collector.collect(
                    model,
                    a.games_per_iter,
                    gamma=a.gamma,
                    lam=a.lam,
                    device=a.device,
                    seed=a.seed + it,
                )
            else:
                buf = collect_rollout(
                    model,
                    deck,
                    a.games_per_iter,
                    opponent=a.opponent,
                    gamma=a.gamma,
                    lam=a.lam,
                    device=a.device,
                    seed=a.seed + it,
                )
            m = ppo_update(model, opt, buf, ppo_cfg, device=a.device)
            if adaptive_ent:  # hold mean entropy at the setpoint by tuning ent_coef
                ppo_cfg.ent_coef = adapt_ent_coef(
                    ppo_cfg.ent_coef, m["entropy"], a.target_entropy, gain=a.ent_gain, lo=a.ent_coef
                )
            dt = time.time() - t0
            stop = "*" if m.get("stopped_kl") else ""
            line = (
                f"it {it:>3}  N={len(buf):>5}  pg={m['pg_loss']:+.3f} vf={m['vf_loss']:.3f} "
                f"ent={m['entropy']:.3f} entc={ppo_cfg.ent_coef:.4f} kl={m['approx_kl']:+.3f}{stop} "
                f"clip={m['clipfrac']:.2f} upd={int(m.get('updates', 0))} lr={cur_lr:.1e}  {dt:.1f}s"
            )

            # P3.3 gated promotion: beat the frozen last-best by the threshold to promote.
            if a.gate and (it % a.gate_every == 0 or it == a.iters):
                assert frozen_best is not None
                r = play_match(model, frozen_best, deck, a.gate_games, device=a.device, seed=a.seed)
                gate_wr = r["winrate"]
                line += f"  | gate vs best {gate_wr * 100:.1f}%(n={r['n']})"
                if gate_wr == gate_wr and gate_wr > a.gate_threshold:  # not NaN
                    frozen_best = copy.deepcopy(model).eval()
                    best = gate_wr
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": cfg.__dict__,
                            "gate_wr": best,
                            "iter": it,
                        },
                        a.out / "best.pt",
                    )
                    line += "  [PROMOTED]"

            if it % a.eval_every == 0 or it == a.iters:
                evals = []
                for opp in eval_opps:
                    r = quick_eval(model, deck, opp, a.eval_games, device=a.device, seed=a.seed)
                    evals.append(f"{opp} {r['winrate'] * 100:.1f}%(n={r['n']})")
                line += "  | eval: " + "  ".join(evals)
                # When not gating, keep the legacy vs-baseline best.pt promotion.
                if not a.gate:
                    wr = quick_eval(
                        model, deck, eval_opps[0], a.eval_games, device=a.device, seed=a.seed
                    )
                    if wr["winrate"] == wr["winrate"] and wr["winrate"] > best:  # not NaN
                        best = wr["winrate"]
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "cfg": cfg.__dict__,
                                "winrate": best,
                                "iter": it,
                            },
                            a.out / "best.pt",
                        )
                        line += f"  [saved best={best * 100:.1f}%]"
            print(line, flush=True)
    finally:
        if collector is not None:
            collector.close()

    torch.save(
        {"model": model.state_dict(), "cfg": cfg.__dict__, "iter": a.iters}, a.out / "last.pt"
    )
    metric = "gate vs best" if a.gate else f"in-loop vs {eval_opps[0]}"
    print(f"\ndone. best {metric} = {best * 100:.1f}%  ->  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
