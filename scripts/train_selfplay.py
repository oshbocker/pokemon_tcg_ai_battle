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


def _load_ckpt_net(path: Path, device: str) -> PtcgNet:
    """Load a frozen ``{model, cfg}`` checkpoint into an eval-mode, grad-free net.

    Mirrors the loader in `eval_harness._build_model_agent`; used for external
    `--league-checkpoint` opponents (any size — the net is built from its own cfg)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["cfg"]) if isinstance(ckpt.get("cfg"), dict) else ModelConfig()
    net = PtcgNet(cfg).to(device).eval()
    net.load_state_dict(ckpt["model"])
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _short(spec: str) -> str:
    """Human label for a league spec (drop the ``kaggle:``/``model:`` prefix)."""
    for pre in ("kaggle:", "model:"):
        if spec.startswith(pre):
            return spec[len(pre) :]
    return spec


def gate_pool_score(
    model: PtcgNet,
    frozen_best: PtcgNet,
    pool: dict[str, PtcgNet],
    deck: list[int],
    fixed_opps: list[tuple[str, float]],
    *,
    games: int,
    device: str,
    seed: int,
    w_self: float,
    w_past: float,
    past_sample: int,
    ext_models: dict[str, PtcgNet] | None = None,
    ext_decks: dict[str, list[int]] | None = None,
) -> tuple[float, list[tuple[str, float, float]]]:
    """Weighted pool win-rate of ``model`` across the training league.

    The gate opponent set mirrors what we train against:
      * the frozen last-best (``self``, weight ``w_self``) and a sample of recent past
        checkpoints (``past`` bucket, combined weight ``w_past``) — both via `play_match`
        on the mirror deck, preserving the "don't regress vs your past self" signal;
      * the fixed manifest agents (kaggle/heuristic/random, their own weights) — via
        `quick_eval`, each piloting its own deck (side-swapped, deck-correct).

    Returns ``(weighted_winrate, breakdown)`` where breakdown is a list of
    ``(label, winrate, weight)`` per opponent bucket (NaN win-rates are reported but
    dropped from the aggregate)."""
    parts: list[tuple[str, float, float]] = []

    # self bucket: current net vs the frozen last-best (anti-regression signal).
    r = play_match(model, frozen_best, deck, games, device=device, seed=seed)
    parts.append(("self", r["winrate"], w_self))

    # past bucket: recent frozen checkpoints, aggregated into one weighted entry.
    past_ids = list(pool)[-past_sample:] if past_sample > 0 else []
    past_wrs = [
        play_match(model, pool[pid], deck, games, device=device, seed=seed)["winrate"]
        for pid in past_ids
    ]
    past_valid = [w for w in past_wrs if w == w]  # not NaN
    if past_valid:
        parts.append(("past", sum(past_valid) / len(past_valid), w_past))

    # fixed agents from the manifest (or flag fallback): each pilots its own deck.
    # A `model:<id>` entry is an external frozen checkpoint (e.g. the trained
    # Archaludon best.pt in the Alakazam league) — score it via an asymmetric
    # `play_match` on ITS deck; everything else runs through `quick_eval`.
    for spec, w in fixed_opps:
        if spec.startswith("model:"):
            mid = spec[len("model:") :]
            r = play_match(
                model,
                (ext_models or {})[mid],
                deck,
                games,
                device=device,
                seed=seed,
                deck_b=(ext_decks or {}).get(spec),
            )
        else:
            r = quick_eval(model, deck, spec, games, device=device, seed=seed)
        parts.append((_short(spec), r["winrate"], w))

    num = sum(w * wr for _, wr, w in parts if wr == wr)
    den = sum(w for _, wr, w in parts if wr == wr)
    score = num / den if den else float("nan")
    return score, parts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--size", default="small", choices=list(SIZE_BANDS))
    ap.add_argument(
        "--deck",
        type=Path,
        default=None,
        help="trainee deck CSV (default: agent/deck.csv). e.g. agent/decks/archaludon.csv",
    )
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
    # P3.4 league (dist + --opponent self): per-game opponent from a pool — fixes the
    # pure-self-play collapse. Weights are relative; 0 omits that opponent.
    ap.add_argument("--league", action="store_true", help="train vs a league, not pure self-play")
    ap.add_argument("--pool-size", type=int, default=4, help="max past checkpoints in the pool")
    ap.add_argument("--snapshot-every", type=int, default=5, help="iters between pool snapshots")
    ap.add_argument("--w-self", type=float, default=1.0)
    ap.add_argument("--w-past", type=float, default=2.0, help="total weight for past checkpoints")
    ap.add_argument("--w-heuristic", type=float, default=1.0)
    ap.add_argument("--w-random", type=float, default=0.5)
    ap.add_argument("--league-kaggle", default=None, help="kaggle_agents/<name> to add to the pool")
    ap.add_argument("--w-kaggle", type=float, default=1.0)
    ap.add_argument(
        "--opp-manifest",
        default="agent/opponents/mixed_pool.json",
        help="JSON opponent pool (agent+deck+weight); drives all fixed/Kaggle opponents. "
        "Empty string disables (fall back to the --w-heuristic/--w-random/--league-kaggle flags).",
    )
    ap.add_argument(
        "--league-checkpoint",
        action="append",
        default=None,
        metavar="PATH:WEIGHT[:DECK_CSV[:LABEL]]",
        help="external frozen checkpoint as a league opponent (e.g. the trained "
        "Archaludon best.pt in the Alakazam league). PATH is a {model,cfg} .pt; WEIGHT "
        "is its sampling weight (keep LOW to avoid over-exposing a weak trainee); DECK_CSV "
        "is the deck it pilots (default: mirror the trainee); LABEL names it in the gate "
        "breakdown (default: the checkpoint's run-dir name). Added to BOTH the training "
        "league and the pool gate; never evicted from the pool. Repeatable.",
    )
    ap.add_argument(
        "--gate", action="store_true", help="promote best.pt by gating vs frozen last-best"
    )
    ap.add_argument(
        "--gate-vs",
        default="mirror",
        choices=["mirror", "pool"],
        help="gate opponent set: 'mirror' = frozen best only (legacy); 'pool' = weighted "
        "win-rate across the training league (self/past + manifest agents)",
    )
    ap.add_argument("--gate-every", type=int, default=5)
    ap.add_argument(
        "--gate-games",
        type=int,
        default=200,
        help="side-swapped games PER opponent bucket (mirror: total vs frozen best)",
    )
    ap.add_argument(
        "--gate-threshold",
        type=float,
        default=0.55,
        help="mirror gate: win-rate to promote. pool gate: absolute floor (0=off), "
        "on top of the relative 'must beat last-promoted pool score' rule",
    )
    ap.add_argument(
        "--gate-past", type=int, default=2, help="pool gate: recent past checkpoints to sample"
    )
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
    deck = load_deck(a.deck)
    eval_opps = [o.strip() for o in a.eval_opponents.split(",") if o.strip()]
    a.out.mkdir(parents=True, exist_ok=True)
    lr_final = a.lr if a.lr_final is None else a.lr_final
    ent_final = a.ent_coef if a.ent_final is None else a.ent_final
    adaptive_ent = a.target_entropy > 0  # controller vs. fixed decay schedule

    # P3.1/P3.4: persistent distributed worker pool (built once, reused every iter).
    # The league (per-game opponent) is the pure-self-play collapse fix.
    use_league = a.league and a.opponent == "self"
    pool: dict[str, PtcgNet] = {}  # frozen past checkpoints (id -> model)
    collector = None
    league = None

    # Deck-agnostic opponent pool (DATA): the manifest drives every fixed/Kaggle
    # opponent + its own deck, so adding/swapping an archetype is a config change.
    # When a manifest is active it supersedes the per-flag heuristic/random/kaggle
    # weights (self/past stay flag-driven). self/model opponents mirror the trainee.
    manifest_extra: list[tuple[str, float]] = []
    manifest_decks: dict[str, list[int]] = {}
    if use_league and a.opp_manifest:
        from ptcg_battle.opponents import load_manifest, manifest_to_league_args

        manifest_extra, manifest_decks = manifest_to_league_args(load_manifest(a.opp_manifest))

    # External frozen checkpoints as league opponents (--league-checkpoint). Each is
    # merged into the league's model set (never evicted) and added to the manifest mix
    # + gate as a `model:<label>` opponent piloting its own deck.
    ext_models: dict[str, PtcgNet] = {}
    if use_league and a.league_checkpoint:
        from ptcg_battle.opponents import read_deck

        for entry in a.league_checkpoint:
            fields = entry.split(":")
            ckpt_path, weight = fields[0], float(fields[1])
            deck_csv = fields[2] if len(fields) > 2 and fields[2] else None
            label = (
                fields[3]
                if len(fields) > 3 and fields[3]
                else (Path(ckpt_path).parent.name or Path(ckpt_path).stem)
            )
            spec = f"model:{label}"
            ext_models[label] = _load_ckpt_net(Path(ckpt_path), a.device)
            manifest_extra.append((spec, weight))
            if deck_csv:
                manifest_decks[spec] = read_deck(deck_csv)

    manifest_on = bool(manifest_extra)
    lw_heuristic = 0.0 if manifest_on else a.w_heuristic
    lw_random = 0.0 if manifest_on else a.w_random
    lkaggle = None if manifest_on else a.league_kaggle

    if a.collector == "dist":
        from ptcg_battle.dist_collector import (
            DistributedCollector,
            League,
            build_league,
            league_fixed_specs,
        )

        if use_league:
            fixed = league_fixed_specs(
                w_heuristic=lw_heuristic,
                w_random=lw_random,
                kaggle=lkaggle,
                w_kaggle=a.w_kaggle,
                extra=manifest_extra,
            )
        elif a.opponent != "self":  # single fixed-opponent training (legacy)
            fixed = [a.opponent] if a.opponent in ("random", "first", "heuristic") else []
            league = League(mix=[(a.opponent, 1.0)])
        else:  # pure self-play
            fixed = []
        collector = DistributedCollector(
            deck, n_workers=a.workers, fixed_specs=fixed, max_steps=4000
        )

    # P3.3: frozen last-best opponent for gated promotion (the anti-collapse net).
    frozen_best = copy.deepcopy(model).eval() if a.gate else None

    # Pool gate: fixed agent opponents = the training league (manifest, else flags).
    gate_vs_pool = a.gate and a.gate_vs == "pool"
    gate_fixed: list[tuple[str, float]] = []
    if gate_vs_pool:
        gate_fixed = list(manifest_extra)
        if not gate_fixed and a.opp_manifest:
            from ptcg_battle.opponents import load_manifest, manifest_to_league_args

            gate_fixed, _ = manifest_to_league_args(load_manifest(a.opp_manifest))
        if not gate_fixed:  # flag fallback (no manifest active)
            if a.w_heuristic > 0:
                gate_fixed.append(("heuristic", a.w_heuristic))
            if a.w_random > 0:
                gate_fixed.append(("random", a.w_random))
            if a.league_kaggle and a.w_kaggle > 0:
                gate_fixed.append((f"kaggle:{a.league_kaggle}", a.w_kaggle))

    deck_name = (a.deck or (REPO / "agent" / "deck.csv")).name
    if use_league and manifest_on:
        opp_desc = "  league: self={} past={}(≤{}) + manifest[{}]".format(
            a.w_self,
            a.w_past,
            a.pool_size,
            ", ".join(f"{s}={w:g}" for s, w in manifest_extra),
        )
    elif use_league:
        opp_desc = (
            f"  league: self={a.w_self} past={a.w_past}(≤{a.pool_size}) "
            f"heur={a.w_heuristic} rand={a.w_random}"
            + (f" kaggle:{a.league_kaggle}={a.w_kaggle}" if a.league_kaggle else "")
        )
    else:
        opp_desc = ""
    print(
        f"train  size={a.size}({total / 1e6:.1f}M, non-emb {nonemb / 1e6:.1f}M)  "
        f"option_rank={cfg.use_option_rank}  opponent={a.opponent}  deck={deck_name}  "
        f"device={a.device}\n"
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
        + opp_desc
        + (
            (
                "  gate[pool"
                + (f"|floor>{a.gate_threshold:.0%}" if a.gate_threshold > 0 else "")
                + f"]/{a.gate_games}g every {a.gate_every} vs "
                + ", ".join(["self", "past", *(_short(s) for s, _ in gate_fixed)])
                if gate_vs_pool
                else f"  gate[mirror]>{a.gate_threshold:.0%}/{a.gate_games}g every {a.gate_every}"
            )
            if a.gate
            else ""
        )
        + "\n"
    )
    best = -1.0
    best_pool = -1.0  # last-promoted checkpoint's weighted pool score (relative gate)
    try:
        for it in range(1, a.iters + 1):
            frac = (it - 1) / max(1, a.iters - 1)
            cur_lr = _lerp(a.lr, lr_final, frac)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            if not adaptive_ent:  # fixed decay; adaptive coef is updated after the step
                ppo_cfg.ent_coef = _lerp(a.ent_coef, ent_final, frac)

            if use_league:  # rebuild the league each iter against the current pool
                league = build_league(
                    pool,
                    w_self=a.w_self,
                    w_past=a.w_past,
                    w_heuristic=lw_heuristic,
                    w_random=lw_random,
                    kaggle=lkaggle,
                    w_kaggle=a.w_kaggle,
                    extra=manifest_extra,
                    opp_decks=manifest_decks,
                    ext_models=ext_models,
                )

            t0 = time.time()
            if collector is not None:
                buf = collector.collect(
                    model,
                    a.games_per_iter,
                    league=league,
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
            if use_league and (it % a.snapshot_every == 0):  # add a frozen "past me" to the pool
                snap = copy.deepcopy(model).eval()
                for p in snap.parameters():
                    p.requires_grad_(False)
                pool[f"it{it}"] = snap
                while len(pool) > a.pool_size:  # evict oldest
                    del pool[next(iter(pool))]
            dt = time.time() - t0
            stop = "*" if m.get("stopped_kl") else ""
            lg = f" pool={len(pool)}" if use_league else ""
            line = (
                f"it {it:>3}  N={len(buf):>5}  pg={m['pg_loss']:+.3f} vf={m['vf_loss']:.3f} "
                f"ent={m['entropy']:.3f} entc={ppo_cfg.ent_coef:.4f} kl={m['approx_kl']:+.3f}{stop} "
                f"clip={m['clipfrac']:.2f} upd={int(m.get('updates', 0))} lr={cur_lr:.1e}{lg}  {dt:.1f}s"
            )

            # P3.3 gated promotion: crown best.pt when the current net clears the gate.
            if a.gate and (it % a.gate_every == 0 or it == a.iters):
                assert frozen_best is not None
                if gate_vs_pool:
                    # Weighted win-rate across the training league (self/past + manifest
                    # agents). Relative gate: promote when it beats the last-promoted pool
                    # score, with an optional absolute floor (--gate-threshold).
                    score, parts = gate_pool_score(
                        model,
                        frozen_best,
                        pool,
                        deck,
                        gate_fixed,
                        games=a.gate_games,
                        device=a.device,
                        seed=a.seed,
                        w_self=a.w_self,
                        w_past=a.w_past,
                        past_sample=a.gate_past,
                        ext_models=ext_models,
                        ext_decks=manifest_decks,
                    )
                    breakdown = " | ".join(
                        f"{lbl} {wr * 100:.0f}" for lbl, wr, _w in parts if wr == wr
                    )
                    line += f"  | gate pool {score * 100:.1f}% [{breakdown}]"
                    promote = (
                        score == score  # not NaN
                        and score > best_pool
                        and (a.gate_threshold <= 0 or score >= a.gate_threshold)
                    )
                    if promote:
                        frozen_best = copy.deepcopy(model).eval()
                        best = best_pool = score
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "cfg": cfg.__dict__,
                                "gate_pool_score": best_pool,
                                "gate_vs": "pool",
                                "iter": it,
                            },
                            a.out / "best.pt",
                        )
                        line += "  [PROMOTED]"
                else:
                    r = play_match(
                        model, frozen_best, deck, a.gate_games, device=a.device, seed=a.seed
                    )
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
    metric = (
        ("gate pool" if gate_vs_pool else "gate vs best")
        if a.gate
        else f"in-loop vs {eval_opps[0]}"
    )
    print(f"\ndone. best {metric} = {best * 100:.1f}%  ->  {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
