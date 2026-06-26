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

## Key findings

1. **Throughput scales ~linearly with *physical* x86 cores, then flatlines.**
   1→2→4 workers = 4.8K→9.4K→18.1K dec/s (near-perfect linear); hyperthreads add
   nothing and slightly hurt. **Rule of thumb on this box: ~4.5K dec/s per
   physical core (engine only).** A modern server core (3–4 GHz, better IPC)
   should do meaningfully more — likely 2–3× per core. ⇒ When renting, **buy
   physical x86 core count**, and set `--procs ≈ physical cores`.

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
