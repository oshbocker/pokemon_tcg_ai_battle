# Calibrating `--gate-games` for the pool promotion gate

How we pick the `--gate-games` value for `scripts/train_selfplay.py --gate-vs pool`.
**Re-run this derivation whenever the training/opponent pool changes** (a different
set of manifest agents, or different weights, changes `k_eff` below and therefore the
right number of games). Locked value as of 2026-06-30: **`--gate-games 40`**.

## What the pool gate estimates

`--gate-vs pool` (`gate_pool_score` in `scripts/train_selfplay.py`) promotes `best.pt`
on a single scalar: the **weighted-mean win-rate** of the current net across the whole
training league —

```
score = Σ wᵢ·wrᵢ / Σ wᵢ
```

Each per-opponent win-rate `wrᵢ` is a binomial estimate from `n = gate-games` games
(`play_match` for self/past on the mirror deck; `quick_eval` for the fixed agents on
their own decks; both serial — the engine is a global singleton, so gate games do NOT
use the 48-worker collector). The variance of the scalar we actually gate on is:

```
Var(score) = Σ wᵢ²·pᵢ(1−pᵢ) / [ (Σwᵢ)²·n ]   ≈  p(1−p) / (n·k_eff)

k_eff = (Σwᵢ)² / Σwᵢ²        # effective number of opponents (weight participation ratio)
```

**Key consequence:** averaging over opponents shrinks the aggregate's sampling noise by
~√k_eff. Between-opponent spread is *signal*, not variance. So a more diverse gate is
*less* noisy at fixed per-opponent `n` — you do **not** multiply games by the number of
opponents. Naive `gate-games × |pool|` is wrong.

## The binding constraint is the breakdown, not the aggregate

The old *mirror* gate used 120 games vs one frozen best → aggregate SE ≈ ±4.6%, and
drove promotions fine. Any reasonable per-opponent `n` already beats that on the
aggregate (see table), because the √k_eff factor buys you margin. So the aggregate is
**not** what sets `gate-games`.

What sets it is the printed per-opponent breakdown
(`gate pool NN% [self.. | starmie.. | alakazam.. | ..]`) — a real diagnostic for
spotting a genuinely bad matchup. Below ~40 games each cell degrades to noise
(`n=16` → ±12.5% SE, ~±25% CI — uninterpretable). Pick `n` so the breakdown stays
legible; the aggregate then comes along for free.

## Current pool (2026-06-30) → `k_eff ≈ 7.2`

Alakazam gate buckets and weights: `self`, `past`, the manifest agents
(archaludon-rule, starmie, dragapult, romanrozen, heuristic, random), plus the
low-weight Archaludon `best.pt` (Option A `model:` opponent) ≈ **9 buckets**,
weights ≈ `[1.0, 2.0, 1.5, 1.5, 1.5, 1.0, 0.5, 0.3, 0.5]`:

```
Σwᵢ  = 9.8      Σwᵢ² = 13.34      k_eff = 9.8² / 13.34 ≈ 7.2
```

At p≈0.5: per-opponent SE = 0.5/√n; aggregate SE = per-opponent SE / √k_eff.

| `--gate-games` | per-opponent SE | aggregate score SE | total games |
|---|---|---|---|
| 120 (old mirror default) | ±4.6% | ±1.7% | ~1,080 |
| **40 (locked)** | ±7.9% | **±2.9%** | **~360** |
| 30 | ±9.1% | ±3.4% | ~270 |
| 16 (matches old aggregate) | ±12.5% | ±4.6% | ~150 |

## Decision: `--gate-games 40`

- Aggregate score SE ≈ **±2.9%** — tighter than the old mirror gate (±4.6%), so
  promotions are at least as reliable.
- Per-opponent breakdown stays legible at ±8% SE (~±15% CI) — enough to see a clearly
  losing matchup.
- Total ≈ **360 serial games** — a third of the naive `120 × 9 = 1,080`, and only ~3×
  the single 120-game past-checkpoint eval. The modest bump over 120 total is justified
  by breakdown readability, not aggregate need.
- Don't go below ~30 or the per-opponent line stops being worth printing.

## Recalibrating when the pool changes

1. List the gate buckets and their weights (`self`, `past`, each manifest agent, any
   Option-A `model:` checkpoints).
2. Compute `k_eff = (Σwᵢ)² / Σwᵢ²`.
3. Pick `n` so per-opponent SE = `0.5/√n` keeps the breakdown legible (target ≈ ±8%,
   i.e. `n ≈ 40`); confirm aggregate SE = `(0.5/√n)/√k_eff` beats the ±4.6% the old
   gate promoted on.
4. Update the locked value and this table.
