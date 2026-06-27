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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from ptcg_battle.model import SIZE_BANDS, ModelConfig, PtcgNet, param_counts  # noqa: E402
from ptcg_battle.ppo import (  # noqa: E402
    PPOConfig,
    collect_rollout,
    load_deck,
    ppo_update,
    quick_eval,
    set_seed,
)

REPO = Path(__file__).resolve().parents[1]


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
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.01)
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
    ppo_cfg = PPOConfig(epochs=a.epochs, minibatch=a.minibatch, ent_coef=a.ent_coef, lr=a.lr)
    deck = load_deck()
    eval_opps = [o.strip() for o in a.eval_opponents.split(",") if o.strip()]
    a.out.mkdir(parents=True, exist_ok=True)

    print(
        f"train  size={a.size}({total / 1e6:.1f}M, non-emb {nonemb / 1e6:.1f}M)  "
        f"option_rank={cfg.use_option_rank}  opponent={a.opponent}  device={a.device}\n"
        f"       iters={a.iters} games/iter={a.games_per_iter} epochs={a.epochs} "
        f"mb={a.minibatch} lr={a.lr} gamma={a.gamma} ent={a.ent_coef}\n"
    )
    best = -1.0
    for it in range(1, a.iters + 1):
        t0 = time.time()
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
        dt = time.time() - t0
        line = (
            f"it {it:>3}  N={len(buf):>5}  pg={m['pg_loss']:+.3f} vf={m['vf_loss']:.3f} "
            f"ent={m['entropy']:.3f} kl={m['approx_kl']:+.3f} clip={m['clipfrac']:.2f}  "
            f"{dt:.1f}s"
        )
        if it % a.eval_every == 0 or it == a.iters:
            evals = []
            for opp in eval_opps:
                r = quick_eval(model, deck, opp, a.eval_games, device=a.device, seed=a.seed)
                evals.append(f"{opp} {r['winrate'] * 100:.1f}%(n={r['n']})")
            line += "  | eval: " + "  ".join(evals)
            wr = quick_eval(model, deck, eval_opps[0], a.eval_games, device=a.device, seed=a.seed)
            if wr["winrate"] == wr["winrate"] and wr["winrate"] > best:  # not NaN
                best = wr["winrate"]
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg.__dict__, "winrate": best, "iter": it},
                    a.out / "best.pt",
                )
                line += f"  [saved best={best * 100:.1f}%]"
        print(line, flush=True)

    torch.save(
        {"model": model.state_dict(), "cfg": cfg.__dict__, "iter": a.iters}, a.out / "last.pt"
    )
    print(f"\ndone. best in-loop vs {eval_opps[0]} = {best * 100:.1f}%  ->  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
