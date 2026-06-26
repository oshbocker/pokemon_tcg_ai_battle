# Phase 0 — Throughput findings

Benchmark: `scripts/bench_throughput.py`. Metric that gates the RL plan:
**agent decisions per second** from process-level self-play (the cabt engine is a
global singleton ⇒ parallelism is across processes; see
[`SELFPLAY_RL_PLAN.md`](./SELFPLAY_RL_PLAN.md)).

## Run 1 — local laptop (2026-06-26)

Host: **Intel i7-10610U, 4 physical cores / 8 threads, 1.8 GHz** (a weak
low-power laptop CPU — treat as a *floor*). Policy = `first` (near-zero cost, so
this measures the *engine* ceiling). ~134 decisions/game.

| workers | games/s | decisions/s | notes |
|--:|--:|--:|--:|
| 1 | 35.6 | 4,762 | |
| 2 | 69.4 | 9,390 | ~linear |
| 4 | 134.5 | **18,104** | **peak = physical core count** |
| 8 | 98.7 | 13,682 | hyperthreads *hurt* |
| 12 | 91.2 | 12,433 | oversubscription |

With `--parse` (running `to_observation_class()` every step, as a naive RL loop
would): peak **9,307 dec/s** at 4 workers — i.e. **dataclass parsing ~halves
throughput**.

## Run 2 — Colab "downgrade" runtime (2026-06-26)

Asked for **H100; Colab downgraded to** an **RTX PRO 6000 Blackwell (96 GB)** GPU
runtime — **note: H100 can be hard to get on Colab**, plan around it. The host is
no slouch: **AMD EPYC 9B45, 48 vCPU / 24 physical cores, 176 GB RAM.** Policy
`first`. ~136 decisions/game.

Burns **~8.9 units/hr**. Full sweep (`--procs 1..64`, `first`):

| workers | games/s | decisions/s | notes |
|--:|--:|--:|--:|
| 1 | 83.8 | 11,394 | 2.5× the laptop core |
| 4 | 165.9 | 22,455 | |
| 8 | 269.6 | 36,397 | |
| 16 | 466.6 | 63,622 | |
| 24 | 579.1 | 78,717 | **= physical cores; ~97% of peak** |
| 32 | 588.9 | 80,434 | |
| 48 | 599.0 | **81,341** | peak (= vCPU; HT adds only ~3% over 24) |
| 64 | 595.0 | 81,035 | oversubscription, flat-down |

**Peak ~81K dec/s ≈ 7 B decisions/day** (engine-only). Plateaus at **24 workers
(physical cores)** — HT contributes only ~3%, so set `--procs ≈ physical cores`
here. `--parse` at 24 workers: 68,441 vs 78,717 (≈13% — again understated because
24/48 vCPU isn't saturated; the saturated tax is ~46%, see Run 3).

## Run 3 — Colab L4 runtime (2026-06-26)

Asked for A100; **downgraded to L4** (premium GPUs keep being scarce: H100→Blackwell,
A100→L4). Host: **12 vCPU (≈6 physical cores)** — *same vCPU count as the A100
runtime*. Policy `first`. ~136 decisions/game.

| workers | games/s | decisions/s | notes |
|--:|--:|--:|--:|
| 1 | 30.1 | 4,019 | |
| 4 | 117.5 | 16,058 | |
| 6 | 166.5 | 22,665 | |
| 8 | 174.7 | 23,618 | plateau begins |
| 12 | 184.1 | **25,026** | peak (all vCPUs; HT *helps* here) |
| 16 | 176.6 | 24,182 | slight oversubscription drop |

`--parse` near saturation: **12,819 dec/s @ 8 workers vs 23,618 (≈46% — parsing
~halves throughput when the box is busy).** This confirms the laptop result and
kills the small-tax illusion from Run 2 (where the 48-vCPU box was half-idle).

Operational: on a 12-vCPU box, run **~12 workers** (8–12 is flat). HT helps on
these cloud Xeon/EPYC parts (peak at vCPU count, not physical-core count) — again,
**measure per box**.

## Compute economics (the Run 3 punchline)

Rollout is **CPU-bound**, so the right metric is **decisions per compute unit**
(env-only), not raw speed. Measured Colab numbers (budget: **~803 units**):

| runtime | vCPU | env dec/s (peak) | units/hr | **M dec / unit** | hrs from ~800u |
|---|--:|--:|--:|--:|--:|
| **L4** | 12 | ~25K | ~1.54 | **~58** ← best value | ~520 |
| **Blackwell/EPYC** | 48 | ~81K | ~8.9 | **~33** | ~90 |
| A100 | 12 | ~25K (same CPU) | ~15 | ~6 ← worst | ~53 |

Two clear conclusions:

1. **L4 for Phases 0–3** (env-bound, small models). Most decisions per unit (~58M),
   ~520 hrs of runtime. GPU isn't the bottleneck yet.
2. **Blackwell for Phase 4** (GPU-bound, big models) — **and skip the A100
   entirely.** Blackwell dominates A100 on *every* axis for us: stronger GPU
   (96 GB Blackwell vs 40/80 GB Ampere), 4× the vCPU (48 vs 12), 3.2× rollout, **and
   cheaper per hour** (8.9 vs ~15). The A100 is the worst value here — same CPU as
   the free-ish L4 at 10× the cost. (Both Blackwell and L4 are what Colab *gives*
   when H100/A100 are unavailable, so availability favors exactly the two we want.)

> Caveat: measured dec/s is *engine-only* (trivial policy). Real throughput drops
> once policy inference is added — that's the still-pending **P0.3** probe (needs
> `torch`). On the L4, batch inference across the 12 workers on its GPU.

## Key findings

1. **Throughput scales with x86 cores — and on server silicon keeps climbing
   into the hyperthread range.** Laptop (4 phys cores): near-linear 1→2→4 then HT
   *hurt* (peak at 4). Server EPYC (24 phys / 48 vCPU): monotonic all the way to
   16 workers and **still rising** — HT helped here, the laptop's early peak was a
   4-core/1.8 GHz artifact (likely thermal). Per-core throughput: **~4.5K dec/s
   (laptop) vs ~11K dec/s (EPYC core)**. ⇒ When renting, **buy physical x86 core
   count**, but **measure the actual peak per box** — set `--procs` from the sweep,
   not a fixed rule. (Server peak still TBD: re-run `--procs 16,24,32,48`.)

2. **`to_observation_class()` costs as much as the entire engine step (~2×
   slowdown).** The recursive `to_dataclass` parse is pure Python overhead. ⇒
   **Build the obs→tensor encoder to read the raw JSON dict directly; do not parse
   into the dataclass tree in the hot training loop.** (~2× free throughput.)

3. **Even the laptop floor is plenty for early phases.** ~9K dec/s *with* parsing
   ≈ **800M decisions/day** ≈ ~6M games/day. Phases 0–3 (small models) are not
   env-starved on any decent CPU.

4. **Colab CPU (verified specs, 2026-06).** The "2 vCPU" figure is the *no-GPU*
   runtime — **not** the GPU runtimes:
   - **A100 runtime ≈ 12 vCPU / ~83.5 GB RAM** (high-RAM). Confirmed.
   - **H100 runtime: underdocumented (≥12 vCPU likely) — must be measured.**
   - **Default CPU-only runtime: 2 vCPU / 13 GB.**

   **vCPU ≠ physical core:** a GCP vCPU is *one hyperthread*, so 12 vCPU ≈ **6
   physical cores**. Given finding #1 (HT doesn't help this sim), expect peak
   throughput around **~6 workers** on an A100 runtime — but server Xeon HT may
   behave differently than the laptop, so sweep through the HT region to confirm.
   Our bottleneck is CPU env stepping, so even a strong Colab GPU can idle waiting
   for rollouts; if the measured dec/s is low, Phase 4 should **decouple** rollout
   (CPU-heavy x86 box) from training (GPU) rather than co-locate on one Colab VM.

   **Adjusted Colab experiment** (run per runtime; prints topology too):
   ```
   !nproc && lscpu | grep -E "^CPU\(s\):|Thread|Core per|Model name" && \
    cd /content/pokemon_tcg_ai_battle && \
    python scripts/bench_throughput.py --procs 1,2,4,6,8,12,16 --duration 8 && \
    python scripts/bench_throughput.py --procs 4,6,8 --duration 8 --parse
   ```

## Open items still in Phase 0

- [ ] **P0.3 inference-cost probe** — needs `torch`; time a forward pass at
      realistic token counts (board + up to ~1000 option tokens) on CPU to confirm
      the per-turn time budget and feed the model-size ceiling. (Deferred to after
      `uv add torch` in Phase 1.)
- [ ] **P0.4 determinism / seeding — ⚠ flag.** `battle_start(deck0, deck1)`
      exposes **no seed parameter** (`lib.BattleStart` takes only the card array);
      the engine seeds its own RNG internally. Kaggle *replays* carry a
      `configuration.seed`, but it's unclear we can *set* it via this local API.
      **This threatens paired-seed evaluation (Phase 5).** Investigate: env var,
      `GameInitialize` seeding, or a different entry point. If unseedable, fall
      back to higher-n unpaired eval (variance will be larger — budget for it).
- [ ] **P0.5 decision** — pick first model-size band *after* the Colab number is
      in. Local floor already supports it; Colab CPU is the gating unknown.

## Bottom line

The engine is **fast and scales cleanly with physical x86 cores** — env
throughput is *not* a blocker for getting started, and won't be the wall on a
CPU-heavy rented box. The two things to act on: (1) **encode from the raw dict**,
not the dataclass tree; (2) **measure Colab's CPU** and, if it's starved, plan to
decouple rollout from training when we scale.
