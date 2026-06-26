# Preliminary Plan — Self-Play RL for Pokémon TCG (cabt)

Status: **draft for review.** Companion to
[`LESSONS_FROM_ORBIT_WARS.md`](./LESSONS_FROM_ORBIT_WARS.md). The lessons there
are the *why* behind every choice below.

Deadline: final submission **Aug 16, 2026**; today **2026-06-26** → ~7 weeks.

## Guiding principles (carried over from Orbit Wars)

1. **Throughput first.** Decide nothing about model size until we know real
   self-play games/sec. (Lesson 1)
2. **Self-play, not imitation.** A heuristic warm-start is allowed only to escape
   random flailing; the teacher is never the target. (Lesson 2)
3. **Low-level, entity-based observation; pointer action head.** Let the model
   learn the card interactions. (Lessons 3, 4)
4. **Paired-seed, high-n evaluation against a fixed honest suite.** Never act on
   small-sample wins. (Lessons 5, 6)
5. **Stabilize with best-checkpoint gating + a league.** (Lesson 7)
6. **Design for CPU / time-limit / bundle-size from the start.** A model that
   times out on Kaggle is worth zero. (Lesson 9)
7. **Scaffold for agentic development; keep strategy human.** (Lesson 10)

## The honest compute reality (decided)

We will **not** have 4×8 B200 / 15B steps. The plan is **scale-adaptive**: build
the pipeline so model size is the easy lever, then push it as far as our actual
compute + throughput allow. The mistake to avoid is the *opposite* of last time —
do not pre-declare a ceiling; let measured scaling curves tell us when to stop.

**Compute decisions (2026-06):**

- **Phases 0–3 run entirely on Colab (A100/H100 credits)** — pipeline build,
  small-model proof, first self-play runs. Free/owned credits; no cash spend.
- **Phase 4 scaling is gated on demonstrated progress.** Budget is up to
  **~10 × $200–500 (≈ $2K–5K)** of cloud bursts, released **only when each prior
  burst shows real self-play progress with leaderboard results to back it up.**
  No big upfront commitment; each tranche must earn the next. **No fixed
  parameter-count target** — the measured scaling curve + ladder rating decide
  how far we push.
- **The env binary is x86-64 only** (`agent/cg/libcg.so`; no aarch64 build). **Any
  machine we rent or buy must be x86.** This rules out Arm boxes (e.g. NVIDIA DGX
  Spark / GB10) for native env stepping — they'd need slow x86 emulation on the
  throughput-critical CPU path. (DGX Spark reconsidered only as post-competition
  personal infra for local large-LLM work, not for this competition.)
- **Our bottleneck is CPU env throughput, not GPU FLOPs.** When sizing cloud
  instances, prioritize **x86 vCPU count** (for parallel game sims) alongside the
  GPU. Confirm in Phase 0.

---

## Environment facts that shape the design

From recon of `agent/cg/` and `scripts/local_selfplay.py`:

- **API loop:** `battle_start(deck0, deck1)` → `obs`; each step
  `to_observation_class(obs)`, read `oc.current` / `oc.select`, return
  `list[int]` indices into `oc.select.option`; `battle_select(action)` →
  next `obs`; ends when `oc.current.result >= 0` (0/1 winner, 2 draw).
- **Global singleton:** `Battle.battle_ptr` is module-global (`cg/game.py`,
  `cg/sim.py`). **One battle per process.** ⇒ parallelism is **process-level**
  (`multiprocessing`), never threads.
- **x86-64 only:** `agent/cg/libcg.so` is an x86-64 ELF (and `cg.dll` x86
  Windows); no aarch64 build ships. **Every training machine must be x86** — Arm
  boxes (DGX Spark / GB10) can't run the env natively.
- **Speed (baseline):** ~40 games / 10s single-process ≈ **~4 games/s/process**,
  ~50–150 decisions/game. Throughput lever = N processes × this. **Must measure
  for real in Phase 0.**
- **Action = pick indices into `obs.select.option`** (length `minCount..maxCount`,
  no dups). The engine only ever offers legal options ⇒ **no action mask needed**
  — a perfect fit for a pointer head over option tokens.
- **`obs.select.context` (`SelectContext`)** labels the decision (`MAIN`,
  `SWITCH`, `ATTACH_FROM`, `ATTACK`, `SETUP_*`, `COIN_HEAD`, ~49 kinds). Most are
  single-pick (`maxCount == 1`); some are multi-pick.
- **Hidden info:** own hand full; opponent hand = count only; deck order hidden;
  prizes face-down. The state we feed the model is **the observation, not ground
  truth** — this is a POMDP.
- **Reward:** terminal ±1 (win/loss), 0 draw. Win = 6 prizes / opponent deck-out
  / opponent has no active Pokémon.
- **Deck:** fixed 60-card list (`deck.csv`), **not** in the action space. Current
  deck = Mega Lucario ex / Hariyama / Lunatone-Solrock. There is a working
  heuristic agent (`agent/main.py`, ~745 lines, Crustle-aware, optional 1-ply
  search) — our **warm-start sanity opponent**, not our teacher.

---

## Phase 0 — Throughput & feasibility spike ⟵ IN PROGRESS

The make-or-break measurement. Findings logged in
[`PHASE0_THROUGHPUT.md`](./PHASE0_THROUGHPUT.md). Benchmark:
`scripts/bench_throughput.py`.

- [x] **P0.1 Measure raw env throughput.** `bench_throughput.py` sweeps worker
      counts (one `battle_ptr` per process, spawn). Local floor (weak 4-core
      laptop): **~18K decisions/s peak at 4 workers**, ~linear in *physical* cores;
      hyperthreads don't help.
- [x] **P0.2 Decision-rate budget.** ~134 decisions/game; ~9K dec/s *with*
      parsing ≈ **~800M decisions/day** on the laptop floor — ample for Phases
      0–3. **Two action items found:** (a) throughput ∝ physical x86 cores ⇒ buy
      core count when renting; (b) `to_observation_class()` ~halves throughput ⇒
      **encode from the raw dict, not the dataclass tree** (~2× free).
- [ ] **P0.3 Inference-cost probe.** Time a forward pass at realistic token
      counts (board + up to ~1000 options) on CPU. Needs `torch` ⇒ deferred to
      Phase 1 (`uv add torch`).
- [ ] **P0.4 Determinism & seedability — ⚠ open risk.** `battle_start` exposes
      **no seed param**; engine seeds internally. May block paired-seed eval
      (Phase 5). Investigate env var / `GameInitialize` seeding / alternate entry
      point; else fall back to higher-n unpaired eval.
- [ ] **P0.5 Run the benchmark on the actual Colab runtime.** Verified: A100 ≈ 12
      vCPU (≈6 physical cores; H100 ≥12, TBD). vCPU = 1 hyperthread, and HT doesn't
      help this sim ⇒ expect peak ~6 workers — measure to confirm. This number
      decides whether Phase-4 must *decouple* CPU rollout from GPU training. Then
      pick the first model-size band.

**Exit criteria:** Colab dec/s measured, seeding question resolved, first
model-size band chosen. (Local feasibility already confirmed.)

## Phase 1 — Scaffolding & contracts (≈3–4 days, parallel with P0)

Set up the agentic-dev guardrails (Lesson 10) so later code generation stays clean.

- [ ] **P1.1 `prepare` gate.** One command: `ruff` + `pyright` + tests +
      doc-freshness (mirror the winner's `just prepare`). Wire into CLAUDE.md.
- [ ] **P1.2 `rl_research/` as the experiment log** (this dir). Every experiment
      gets a dated entry; dead-ends recorded (our best Orbit Wars habit).
- [ ] **P1.3 Observation/action contract doc** (`docs/rl-obs-action.md`): the
      exact tensor encoding of `Observation` → tokens and `option` → candidate
      tokens. This is the spec agents implement against.
- [ ] **P1.4 Parity/encoding tests:** round-trip a real `obs_dict` through the
      encoder and assert invariants (every legal option maps to exactly one
      candidate token; masks correct; multi-pick handled).
- [ ] **P1.5 Add `torch` to deps** (`uv add torch`), with a CPU/GPU note.

## Phase 2 — Observation/action encoding + model skeleton (≈1 week)

Low-level, entity-based, pointer head (Lessons 3, 4). Start **small** (~1–5M
params) to validate the pipeline before scaling.

- [ ] **P2.1 Entity tokenization.** Tokens for: own/opp **active** & **bench**
      Pokémon (HP, energies, status, stage, tool), **hand** cards (own), **board
      summary** tokens (prizes left each side, deck counts, hand count, turn,
      stadium, who's-deciding), and a learned token per **legal option** in
      `obs.select.option` (encode `OptionType`, `area`, target card/Pokémon
      identity, `attackId`, `number`). Card identity via a learned **card-ID
      embedding** (≈1300 cards) — this is where card knowledge lives, learned not
      hand-coded.
- [ ] **P2.2 Shared transformer trunk** over all tokens (start: ~6–12 blocks,
      d=256–512, like the winner's structure but sized to our compute). Include a
      few learned **summary/scratch tokens** (his trick — a global workspace).
- [ ] **P2.3 Pointer actor head.** Score each option token by attention against a
      "decision" token (`Q·K/√d`), softmax over the *current legal option set*.
      For **multi-pick** decisions (`maxCount > 1`): autoregressive selection or
      independent Bernoulli per option with a count constraint — decide in P2.1
      doc. Legality is free (only legal options are tokens).
- [ ] **P2.4 Critic head.** Win-probability from a value token (sigmoid → value
      in [-1,1]). POMDP ⇒ critic sees only the observation (acceptable; matches
      what the policy sees).
- [ ] **P2.5 Single-process rollout + PPO update** wired end to end on the small
      model. Goal: **learn to beat the random agent decisively**, sanity-checking
      the whole loop before any scale.

## Phase 3 — Self-play training loop (≈1 week)

PPO with the winner's stabilizers (Lesson 7).

- [ ] **P3.1 Distributed rollout collector.** N worker processes each run
      self-play games (one `battle_ptr` each), send trajectories to a learner.
      This is our throughput engine — tune N to the Phase-0 saturation point.
- [ ] **P3.2 PPO learner:** GAE-λ, clipped PG, advantage normalization, entropy
      bonus, value loss. **γ:** start ~0.997 (turn-based, finite horizon) — *not*
      necessarily 1.0; we don't have the winner's 4-player constraint, and γ<1
      avoids his stalling problem (Lesson 8). Reward = terminal ±1.
- [ ] **P3.3 Best-checkpoint gating.** Trainee plays a **frozen last-best**
      opponent; promote on **>~60–70% paired-seed win-rate**. Add small **KL** +
      **value-CE** terms vs the frozen checkpoint for stability.
- [ ] **P3.4 League / past-checkpoint pool** (the winner's #1 regret — do it
      early). Sample opponents from {current self, last-best, a few past
      checkpoints, the heuristic bot}. Guards against non-transitive
      self-overfitting (Lesson 7).
- [ ] **P3.5 (Optional) brief BC warm-start** from the heuristic agent purely to
      skip the random-flailing phase — then RL past it. Skip if P2.5 self-play
      already bootstraps fine.
- [ ] **P3.6 (Optional) potential-based prize-diff shaping** if sparse reward +
      high variance stalls learning. Potential-based only, so optimal policy is
      unchanged (Lesson 8). Treat as an experiment, default off.

## Phase 4 — Scale up (ongoing, gated by Phase 0 *and* by leaderboard progress)

The Bitter Lesson bet, sized to our actual compute (Lesson 1, the central lesson).
**Spend is earned, not pre-committed:** each ~$200–500 cloud burst (x86, GPU +
many vCPUs) is released only after the previous one shows real self-play progress
*and a leaderboard result to back it up*. No fixed parameter-count target.

- [ ] **P4.0 First paid burst trigger.** Only after Phases 0–3 (on free Colab)
      produce an agent that (a) clearly beats the heuristic bot in high-n
      paired-seed eval **and** (b) posts a respectable ladder rating on a real
      submission. That submission is the baseline the scaling sweep must beat.
- [ ] **P4.1 Scaling sweep.** Hold the recipe; grow model size in steps
      (e.g. 5M → 20M → 60M → …). **Plot win-rate vs the fixed eval suite — and
      ladder rating — at each size.** As long as both climb, release the next
      burst and keep scaling. *This is the experiment we never ran in Orbit Wars.*
- [ ] **P4.2 Raise throughput to feed the bigger model:** more x86 processes,
      batched inference (collect decisions across workers into one forward pass),
      pinned buffers, GPU rollout inference if it helps.
- [ ] **P4.3 Stop / pause spend** when the measured curve flattens, the ladder
      stops improving, or we hit the inference/bundle constraint — not on a hunch,
      and never throwing good money after a flat curve.

## Phase 5 — Evaluation harness (continuous, stand up early)

Our strongest Orbit Wars habit (Lesson 6). Build the skeleton in Phase 2.

- [ ] **P5.1 Fixed honest opponent suite:** heuristic agent, mirror, random, and
      the strongest public kernels (Crustle wall, Dragapult — in `outputs/kernels`).
- [ ] **P5.2 Paired-seed, side-swapped, high-n** (start n≥100; Pokémon variance
      is *higher* than Orbit Wars — re-measure the A/A noise floor and set the
      "don't act below this n" threshold from data).
- [ ] **P5.3 Resumable CSV results + a single `eval` command.** Track every
      checkpoint. The eval suite is the source of truth, not training reward.

## Phase 6 — Submission engineering (≈1 week before deadline; design-aware throughout)

Lesson 9 — start considering this in Phase 2, execute near the end.

- [ ] **P6.1 Per-turn time-limit & bundle-size budget** confirmed (re-pull
      competition pages). Bundle = `main.py` + `deck.csv` + `cg/` + model weights.
- [ ] **P6.2 Quantize for CPU inference** (int8 dynamic) and **for file size**
      (NF4-style if needed) — copy the winner's recipe to the extent the size
      forces it.
- [ ] **P6.3 Time-budget fallback** (the winner's trick): if a turn risks
      timeout, fall back to a tiny model or the heuristic agent. Crash-safe
      default to a legal move (the current `agent/main.py` already does this).
- [ ] **P6.4 `local_selfplay.py` validation must pass** (agent vs itself, 0
      errors) before every submission — this is exactly Kaggle's validation
      episode.

---

## Risks & open questions

- **Throughput ceiling (biggest risk).** If process-level parallelism can't get
  us enough decisions/sec, large-scale self-play is off the table and we fall
  back to a smaller model + maybe light search. **Phase 0 decides this.**
- **POMDP + high variance.** Hidden hands and coin flips mean noisier returns and
  harder credit assignment than Orbit Wars. Mitigations: high-n eval, league play,
  possibly recurrent/history encoding if single-observation policies struggle.
- **Deck choice is out of scope of RL** but matters a lot (non-transitive meta).
  Decision: **fix the deck** for the first training run (learn to *play* well);
  treat deck optimization as a separate later pass. Consider training vs a *pool
  of meta decks* as opponents so we don't overfit to the mirror.
- **Multi-pick decisions** (`maxCount > 1`) complicate the pointer head — needs a
  clean spec in P1.3/P2.1.
- **x86-only env binary** — every rented/owned machine must be x86 (no Arm).
  Re-check if organizers ever ship a new engine build.

## Suggested immediate next steps

1. Execute **Phase 0** (throughput spike) on Colab — it's free and gates
   everything (especially the x86 vCPU-count question).
2. In parallel, stand up the **Phase 1** scaffolding and the **Phase 5** eval
   skeleton.
3. Build through **Phases 2–3** on Colab credits → first real submission.

Only then (per **P4.0**) do we open the wallet for the scaling sweep, and only as
far as leaderboard results justify.
