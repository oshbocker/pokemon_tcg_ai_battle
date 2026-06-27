#!/usr/bin/env python
"""A/B the `use_option_rank` feature — does the engine-order prior earn its place?

The engine enumerates `select.option` strong→weak (B1 = option 0 beats random
~88–90%, Kaggle #713608). A pointer head is permutation-invariant, so we expose
the option index as `opt_rank` and gate it with `use_option_rank` (model.py). This
trains both arms under an identical recipe/seeds, then judges them the way we judge
everything else: **high-n, side-swapped, Wilson CIs** — not the noisy in-loop eval.

Three measurements on the best checkpoint of each arm:
  1. vs `random`   2. vs `first` (= B1, the engine-order baseline)   3. head-to-head ON vs OFF.

    uv run --extra rl python scripts/ablate_option_rank.py \
        --seeds 2 --iters 12 --games-per-iter 24 --eval-n 600

Single-machine CPU run is a *directional* read (small model, few seeds); rerun on
the L4 with `--size small --seeds 4` before treating it as settled.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from ptcg_battle.encoding import encode_observation  # noqa: E402
from ptcg_battle.eval_harness import wilson_interval  # noqa: E402
from ptcg_battle.model import SIZE_BANDS, ModelConfig, PtcgNet  # noqa: E402
from ptcg_battle.ppo import (  # noqa: E402
    PPOConfig,
    _import_engine,
    collect_rollout,
    load_deck,
    ppo_update,
    quick_eval,
    set_seed,
)

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"


def train_arm(use_rank: bool, base_cfg: ModelConfig, deck, args, seed: int) -> PtcgNet:
    """Train one arm; return the best-by-in-loop-eval checkpoint (loaded)."""
    set_seed(seed)
    cfg = ModelConfig(**{**base_cfg.__dict__, "use_option_rank": use_rank})
    model = PtcgNet(cfg).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(epochs=args.epochs, minibatch=args.minibatch)
    best_state, best_wr = copy.deepcopy(model.state_dict()), -1.0
    for it in range(1, args.iters + 1):
        buf = collect_rollout(
            model,
            deck,
            args.games_per_iter,
            opponent=args.opponent,
            device=args.device,
            seed=seed * 1000 + it,
        )
        ppo_update(model, opt, buf, ppo_cfg, device=args.device)
        if it % args.eval_every == 0 or it == args.iters:
            wr = quick_eval(model, deck, "random", args.sel_n, device=args.device, seed=999)
            if wr["winrate"] == wr["winrate"] and wr["winrate"] > best_wr:  # not NaN
                best_wr, best_state = wr["winrate"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def head_to_head(model_a: PtcgNet, model_b: PtcgNet, deck, n_games: int, device: str) -> dict:
    """A vs B, greedy, side-swapped. Returns A's wins/losses/draws."""
    prev = Path.cwd()
    os.chdir(AGENT_DIR)
    a_w = b_w = draws = 0
    try:
        to_oc, battle_start, battle_select, battle_finish = _import_engine()
        for g in range(n_games):
            a_seat = g % 2
            obs, result = None, None
            try:
                obs, _ = battle_start(deck, deck)
                if obs is None:
                    continue
                for _ in range(4000):
                    oc = to_oc(obs)
                    cur = oc.current
                    if cur is None:
                        break
                    if cur.result is not None and cur.result >= 0:
                        result = cur.result
                        break
                    m = model_a if cur.yourIndex == a_seat else model_b
                    out = m.act([encode_observation(obs)], sample=False, device=device)[0]
                    obs = battle_select(out["action"])
            finally:
                if obs is not None:
                    with contextlib.suppress(Exception):
                        battle_finish()
            if result is None:
                continue
            if result == 2:
                draws += 1
            elif result == a_seat:
                a_w += 1
            else:
                b_w += 1
    finally:
        os.chdir(prev)
    return {"a_wins": a_w, "b_wins": b_w, "draws": draws}


def pooled_winrate(models: list[PtcgNet], deck, opponent: str, n_each: int, device: str) -> tuple:
    w = ln = 0
    for m in models:
        r = quick_eval(m, deck, opponent, n_each, device=device, seed=4242)
        w += r["wins"]
        ln += r["losses"]
    return w, ln


def fmt(w: int, ln: int) -> str:
    p, lo, hi = wilson_interval(w, w + ln)
    return f"{p * 100:5.1f}%  [{lo * 100:4.1f},{hi * 100:4.1f}]  (n={w + ln})"


def _cmp_props(w1: int, l1: int, w2: int, l2: int) -> tuple[float, float]:
    """Two-proportion difference (p1−p2) and its z-score (normal approx)."""
    import math

    n1, n2 = w1 + l1, w2 + l2
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0
    p1, p2 = w1 / n1, w2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return diff, (diff / se if se > 0 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", default="tiny", choices=list(SIZE_BANDS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--opponent", default="random", choices=["self", "random", "first"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--sel-n", type=int, default=120, help="in-loop games for best-checkpoint pick")
    ap.add_argument("--eval-n", type=int, default=600, help="final games per arm per opponent")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base_cfg = SIZE_BANDS[args.size]
    deck = load_deck()
    print(
        f"option-rank ablation  size={args.size}  seeds={args.seeds}  iters={args.iters} "
        f"games/iter={args.games_per_iter}  opponent={args.opponent}  device={args.device}\n"
        f"final eval n={args.eval_n}/arm/opponent (pooled over seeds)\n"
    )
    t0 = time.time()
    arms: dict[bool, list[PtcgNet]] = {True: [], False: []}
    for seed in range(args.seeds):
        for use_rank in (True, False):
            tag = "ON " if use_rank else "OFF"
            ts = time.time()
            arms[use_rank].append(train_arm(use_rank, base_cfg, deck, args, seed))
            print(f"  trained rank={tag} seed={seed}  ({time.time() - ts:.0f}s)", flush=True)

    n_each = max(1, args.eval_n // args.seeds)
    print("\n=== high-n eval (pooled over seeds, side-swapped, Wilson 95%) ===")
    print(f"{'arm':>10} {'vs random':>26} {'vs first (B1)':>26}")
    rows = {}
    for use_rank in (True, False):
        rw, rl = pooled_winrate(arms[use_rank], deck, "random", n_each, args.device)
        fw, fl = pooled_winrate(arms[use_rank], deck, "first", n_each, args.device)
        rows[use_rank] = (rw, rl, fw, fl)
        tag = "rank ON" if use_rank else "rank OFF"
        print(f"{tag:>10} {fmt(rw, rl):>26} {fmt(fw, fl):>26}")

    print("\n=== head-to-head: rank ON vs rank OFF (paired by seed, side-swapped) ===")
    hw = hl = hd = 0
    for a, b in zip(arms[True], arms[False], strict=True):
        r = head_to_head(a, b, deck, n_each, args.device)
        hw += r["a_wins"]
        hl += r["b_wins"]
        hd += r["draws"]
    _, h_lo, h_hi = wilson_interval(hw, hw + hl)
    print(f"  ON {fmt(hw, hl)}  vs OFF   (draws={hd})")

    # PRIMARY signal = on-distribution (the metric the arms were trained for:
    # win-rate vs `--opponent`). The head-to-head pits two fixed-opponent-trained
    # nets OUT of distribution, so a lopsided H2H is a non-transitivity FLAG, not a
    # feature verdict — report it as such instead of letting it decide.
    on_rw, on_rl = rows[True][0], rows[True][1]
    off_rw, off_rl = rows[False][0], rows[False][1]
    diff, z = _cmp_props(on_rw, on_rl, off_rw, off_rl)
    primary = (
        f"ON is +{diff * 100:.1f}pp vs the {args.opponent} baseline"
        if z > 1.96
        else f"OFF is +{-diff * 100:.1f}pp vs the {args.opponent} baseline"
        if z < -1.96
        else f"no significant on-distribution gap ({diff * 100:+.1f}pp, z={z:.1f})"
    )
    h2h = "ON>OFF" if h_lo > 0.5 else "OFF>ON" if h_hi < 0.5 else "~tie"
    flag = ""
    if (z > 1.96 and h_hi < 0.5) or (z < -1.96 and h_lo > 0.5):
        flag = (
            "  [NON-TRANSITIVITY: H2H disagrees with on-distribution — the actual\n"
            "   question (which feature wins in SELF-PLAY) needs a self-play A/B]"
        )
    print(
        f"\nPRIMARY (on-distribution): {primary}."
        f"\nH2H probe: {h2h} (ON wins {hw / max(1, hw + hl) * 100:.1f}% of decisive)."
        f"{flag}"
        f"\nelapsed {time.time() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
