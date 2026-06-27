# Ablation — the `opt_rank` (engine-order) feature (2026-06-26)

**Question.** The engine enumerates `select.option` strong→weak — returning option
0 ("B1") beats random ~88–90% (Kaggle #713608). Our pointer head is
permutation-invariant, so we expose the option index as `opt_rank` and gate it
with `use_option_rank` ([[PHASE1_RESEARCH]]). Does that prior earn its place, or is
it a heuristic crutch the model should learn past (Lesson 3: the Orbit Wars winner
found masks/bins *harmful*)?

**Method.** `scripts/ablate_option_rank.py`. Both arms (ON / OFF) trained under an
identical recipe + seeds, best checkpoint judged the way we judge everything:
high-n, side-swapped, Wilson CIs. Run: **tiny model, 2 seeds, 12 iters × 24
games, trained vs `random`, eval n≈400/arm (pooled), CPU.**

## Result

| arm | vs `random` | vs `first` (B1) |
|---|---|---|
| **rank ON** | **82.8%** [78.7, 86.1] | **38.5%** [33.9, 43.4] |
| rank OFF | 74.7% [70.2, 78.7] | 32.6% [28.2, 37.3] |

- **On-distribution (the metric the arms were trained for — beat `random`): ON is
  +8.1 pp, z = 2.80 (significant).** vs `first`: ON +5.9 pp, z = 1.75 (directional,
  not significant at this n).
- **Head-to-head ON vs OFF: ON wins only 11.5%** [8.7, 15.0] (OFF dominates).

## Reading — ON wins on-distribution; the H2H is a non-transitivity artifact

The head-to-head *looks* like OFF crushes ON, but it's **out of distribution**:
both arms were trained to beat a fixed weak opponent (`random`), so neither has
ever faced a competent net — pitting them against each other measures a matchup
neither trained for. Pokémon is non-transitive (A>B, B>C, C>A; the plan's central
caution), and a lopsided OOD head-to-head between two narrowly-trained policies is
exactly that, **not** a verdict on the feature. The script now flags this
explicitly rather than letting the H2H decide (its first auto-verdict — "OFF is
better" — was misleading; fixed to make the on-distribution delta primary).

The honest signal is the on-distribution one: **with `opt_rank`, the agent learns
to beat the baselines faster/better** (+8 pp vs random, significant; +6 pp vs B1,
directional). That matches the theory — the engine's ordering is a strong free
prior — and there's no on-distribution evidence it's a harmful crutch (yet).

There *is* a hypothesis worth holding: ON may overfit to "trust the engine order,"
which helps vs weak opponents but could generalise worse against strong ones (a
crutch). The H2H is weakly consistent with that, but too confounded to act on.

## Decision

- **Keep `use_option_rank=True` as the default** — provisional. On-distribution
  evidence + theory favor it; cost is trivial (one small embedding).
- **Definitive test deferred to a self-play A/B on the L4:** `--opponent self
  --size small --seeds 4`, evaluate both arms on the honest suite (random, first,
  heuristic) at high n. Self-play is the real use case and removes the
  train-vs-random confound. If ON shows the "crutch overfits in self-play"
  pattern there, flip the default to OFF.

## Caveats

Tiny model, 2 seeds, CPU, trained vs `random` (not self-play). Directional, not
settled. The late-training PPO collapse (P2.5 note) means we compare
best-checkpoints, not final — fair (same procedure both arms) but adds selection
noise at the in-loop `sel-n`. Rerun bigger on the L4 before treating as final.
