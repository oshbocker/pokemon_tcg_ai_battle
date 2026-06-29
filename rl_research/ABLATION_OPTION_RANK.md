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

---

## Phase-3 definitive A/B (self-play) — harness built, L4 run PENDING (2026-06-27)

The deferred settle is now executable. `scripts/ablate_option_rank_selfplay.py`
trains both arms under the **stabilized Phase-3 recipe** and removes the two
confounds the Phase-2 run flagged:

- **Self-play, not vs-random.** `DistributedCollector` (P3.1) collects `--opponent
  self` for both arms — the real use case, so the eval is on-distribution and the
  H2H is no longer an OOD artifact (both arms have faced competent nets).
- **No collapse.** KL early-stop (`--target-kl`), LR + entropy decay, and
  best-checkpoint **gating vs a frozen last-best** (P3.2/P3.3) keep each arm from
  the P2.5 wobble, so the comparison isn't contaminated by a collapsed final.

Judging is unchanged in spirit — high-n, side-swapped, Wilson CIs — but now runs
through the real eval harness via `model:<path>` agents (P3 task 2): each arm is
scored on the **full honest suite (`random`, `first`/B1, `heuristic`)** plus a
**model-vs-model head-to-head** (ON champion vs OFF opponent). The script prints a
per-opponent two-proportion verdict and a RECOMMENDATION (keep ON / flip to OFF /
inconclusive). Watch for the hypothesized **"crutch overfits in self-play"**
pattern: ON beating weak baselines but losing the H2H and/or `heuristic`.

Run it on the L4 (`notebooks/colab_selfplay.ipynb`, cell 3):

```
python scripts/ablate_option_rank_selfplay.py --size small --workers 12 \
    --seeds 2 --iters 80 --games-per-iter 128 --eval-n 2000 --device cuda
```

**Result:** _final verdict pending a clean re-run (see below)._ A laptop smoke
(`--size tiny --iters 3`) confirms the pipeline runs and emits the verdict; it is
**not** a result (tiny/under-trained → pure noise).

### First L4 run (2026-06-27) — confounded by a collapse; informative anyway

The first `small`, 2-seed L4 run (the original recipe: per-*epoch* KL check,
entropy decayed to ~1e-3) gave a strong in-loop signal *and* exposed a bug:

- **ON runs hot / low-entropy** (~0.01–0.05), vs-random 95–98%, learns fast.
- **OFF runs cool / high-entropy** (~0.24–0.28), vs-random ~70–90%, slower but steady.
- **ON seed 1 collapsed** (~it20): entropy hit ~0.005, then vs-random 98%→45%,
  gate 61%→2%, with a KL spike to ~3.2. Exactly the **"crutch overfits → brittle"**
  pattern hypothesised above — `opt_rank` lets the policy collapse onto the engine
  order, entropy craters, and one over-large update detonates it. (Best-checkpoint
  gating preserved the pre-collapse it15 weights, so nothing un-trained was saved.)

This makes the run's ON arm **under-trained on seed 1** (≈15 useful iters vs OFF's
80) → the A/B is confounded; do **not** read a default off it. But it diagnosed the
real cause: the **per-epoch** KL early-stop was too coarse (one epoch drifted
~0.5–3 KL before the brake fired). Fixes shipped before the re-run:

- **Per-minibatch KL trust region** (`ppo_update`) — bounds each iteration's drift
  to ~`target_kl` (default raised to 0.5, the demonstrated-safe operating point).
- **Entropy floor** — ablation `--ent-decay` default 0.5 (final ~5e-3), so entropy
  no longer decays to the ~0 that preceded the collapse.
- (Also: the pointer-logit `1/√d` fix and a vectorised `act()` for throughput.)

**Re-run** with the fixed recipe on the L4, then record the honest-suite table +
RECOMMENDATION here and set the default. The brittleness of ON is itself evidence
to weigh: even if ON wins vs-random, if it needs the entropy floor to avoid
collapse while OFF is stable without it, that is a mark against the crutch.

### Training regime is now the LEAGUE, not pure self-play (P3.4)

Pure self-play kept collapsing (entropy → 0, sub-random) even after the entropy
controller + LR fixes — a degenerate mutual-best-response is too strong an
attractor for the ON arm. So **both arms now train against a league** (per-game
opponent from `{self, frozen past checkpoints, heuristic, random, the vendored
Kaggle `romanrozen_v10` agent}`; `dist_collector.League`). This is a *better* test
of the feature than pure self-play: the league stresses generalization (you must
beat varied competent opponents, not just out-script a mirror), so a crutch that
overfits the engine order should show up as worse honest-suite / H2H numbers, not
as a training collapse that confounds the comparison. Both arms get the identical
league + best-by-vs-random selection, so the comparison stays fair.

### Sanity gate PASSED — A/B is cleared to run (2026-06-29)

The go/no-go for the A/B (3-seed `small` stability under the league, ~35 iters,
ON arm) **passed on all three seeds**. Every seed showed the controller working
as designed: entropy dipped into the old collapse zone early (min ~0.013–0.022
around it6–7), `entc` ratcheted **up** (0.02 → 0.18–0.28), and entropy recovered
**while vs-random kept climbing** — no seed cratered.

| seed | min `ent` (early) | `entc` peak | `ent` final | vs-random final | gate best |
|---|---|---|---|---|---|
| 0 | 0.022 (it7) | 0.28 | 0.805 | **95.8%** | never promoted |
| 1 | 0.015 (it7) | 0.18 | 0.113 | **93.3%** | 60.8% |
| 2 | 0.013 (it6) | 0.21 | 0.229 | **90.8%** | 64.7% |

vs-random finals cluster 90.8 / 93.3 / 95.8% (all ≫ 75%); KL stayed bounded
(circuit-breaker fired only on early hot iters + a brief seed-1 it12–16 wobble).
This is the exact collapse that killed pure self-play, now caught and reversed —
recipe-v2 + entropy controller + league is validated for `small`. A standalone
40-iter ON league run agrees (ent final 0.34, vs-random 87–90%, vs-`first`/B1
**79%**, gate promoted 3×). Steady-state entropy varies widely across seeds
(0.11 → 0.8); the controller holds the *floor* reliably, not a tight setpoint —
a possible tuning item, not a blocker.

**Still PENDING: the A/B itself** (OFF arm + honest-suite n≈2000 table + H2H).
The sanity only exercises the ON arm, so the crutch hypothesis is unsettled and
`use_option_rank=True` stays the provisional default. Run
`ablate_option_rank_selfplay.py --size small --workers 32 --seeds 2 --iters 50
--games-per-iter 128 --eval-n 2000 --device cuda`, verify both arms' traces are
clean (no collapsed arm), then record the table + RECOMMENDATION here.

**Orthogonal blocker surfaced by these runs: the heuristic wall.** Every run
loses to `heuristic` at ~12–16% and does *not* trend up even as vs-random → 90%
and vs-`first` → 79%. P4.0 gates the first paid burst on beating the heuristic,
so this — not option-rank — is the gate to Phase 4. Levers: raise
`--w-heuristic`/`--w-kaggle`, lower `--w-self`; if opponent pressure doesn't move
~15%, treat it as the scale signal (5.7M capacity-bound vs a competent script).

### A/B run done (2-seed, 80-iter, league) — traces clean, honest table PENDING (2026-06-29)

Ran `ablate_option_rank_selfplay.py --size small --seeds 2 --iters 80 --eval-n
2000`. **Both arms produced clean best-checkpoints — no confounding terminal
collapse** (the first-L4-run defect, where ON s1 detonated, is gone; the league +
recipe-v2 delivered comparable arms). Per-arm training trace:

| arm | entropy behavior | `entc` | vsRand final |
|---|---|---|---|
| ON s0 | dip to 0.014 (it10) → recover → overshoot 0.22 | hit **ceiling 0.30** | 98–100% |
| ON s1 | **sub-random through it20** (43/52/44%), KL fired it10/15, recover, runs to 0.76 | up to 0.16 | 98–99% (best it70) |
| OFF s0 | steady 0.18→0.28, no intervention | **never left floor 0.02** | 95% |
| OFF s1 | steady 0.16–0.24, no intervention | **never left floor 0.02** | 72–76% |

**Two axes, opposite directions — exactly the tie the honest suite must break:**

1. **Stability → favors OFF (mark against the crutch).** ON is brittle: both seeds
   needed the entropy machinery working overtime (s0 `entc` → ceiling; s1
   sub-random for 20 iters with the KL breaker cutting updates short), with wild
   entropy swings (0.014 → 0.76). **ON needs the floor to avoid collapse.** OFF is
   rock-stable — `entc` never left 0.02 for either seed; entropy self-regulated.
   This is the doc's stated "mark against ON."
2. **On-distribution (vsRand, the selection metric) → favors ON.** ON 98–100% both
   seeds; OFF 95% / **72%** (s1 underperformed). Corroborates the Phase-2 +8pp.

**Honest-suite eval HUNG — no table printed.** Root cause (verified by pulling
`gdrive:ptcg_outputs/ablation_sp/`): the honest-eval phase is the *only* path that
runs torch on CPU in many spawned workers (training workers are torch-free → GPU),
and `eval_harness.evaluate()` caps no threads — `cpu_count-1` workers × default
intra-op threads ≈ 130 threads on a 12-vCPU L4 → oversubscription. `heuristic`
matchups (model loses → long games) dominate wall-time; the run crept across two
days via resumable CSVs and stalled in `ON-s1 vs heuristic` at 22/40 chunks (clean,
0 errors → not the max_steps guard; session ended or a native-engine game wedged,
which `max_steps` can't interrupt). The table prints only after *all* matchups, and
`imap_unordered` has no per-chunk timeout → bare header, no rows.

**Recovered ON-arm honest suite from the CSVs** (high-n, side-swapped, pooled):

| opponent | ON win-rate (Wilson 95%) | n |
|---|---|---|
| `random` | 88.8% [87.4, 90.2] | 2000 |
| `first`/B1 | 79.5% [77.7, 81.3] | 2000 |
| `heuristic` | **26.8%** [24.7, 29.1] | 1550 (partial) |

ON beats random + out-plays B1 but still loses to heuristic (26.8%, up from ~15%
at 40 iters — more training helped, wall stands). **OFF arm + H2H never ran**
(no `eval_off_*`/`h2h_*` files) → A/B still **INCOMPLETE**, ON column only.

**Fixes before re-run** (named, `prepare`-safe): (1) `eval_harness.evaluate` — Pool
`initializer` pinning `torch.set_num_threads(1)` + `OMP/MKL_NUM_THREADS=1` (biggest
throughput win); (2) per-chunk timeout in `evaluate` so a wedged native game isn't
fatal; (3) `ablate_option_rank_selfplay.py` lines 285-306 — print each cell as its
`evaluate()` returns (live progress + partial capture); (4) drop `--eval-n` to
~1200 and/or run the eval model agent on cuda.

**Provisional decision: KEEP `use_option_rank=True`, INCOMPLETE.** Don't flip — the
rule is flip only if OFF wins *on-distribution*, and it doesn't (ON wins vsRand;
OFF's stability is a mark against ON, not a win for OFF). Don't call it settled
either — the heuristic/H2H numbers could still reveal "ON beats weak baselines but
loses where it counts." **Finalize rule:** if ON's honest-suite edge over OFF is
<5pp *and* ON loses the H2H or `heuristic`, flip to OFF (brittle crutch not
earning its keep); if ON wins the honest suite by a clear margin, keep ON and
treat the brittleness as floor-managed.
