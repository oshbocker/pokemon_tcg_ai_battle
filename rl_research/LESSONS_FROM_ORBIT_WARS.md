# Lessons from Orbit Wars

Retrospective distilled from two sources:

1. **Our own Orbit Wars repo** (`~/git/orbit_wars`) — final result **~rank 140 / 4212,
   LB 1201.6**, achieved by *forking the public "Producer" planner* and adding
   arena-validated micro-deltas. Our pure-RL track stalled and was abandoned.
2. **The competition winner** — Isaiah Pressman (Tufa Labs),
   [`IsaiahPressman/kaggle-orbit-wars`](https://github.com/IsaiahPressman/kaggle-orbit-wars),
   [write-up](https://www.kaggle.com/competitions/...): a single **200M-parameter
   transformer**, **15B steps of pure self-play PPO**, **~2400 B200-hours**, no
   imitation learning. He won.

The gap between us and him is the whole point of this document. Read it before
designing the Pokémon agent. The short version: **we bet on cleverness under a
compute ceiling we imposed on ourselves; he bet on scale and throughput, and the
Bitter Lesson held.**

---

## The single biggest lesson

> **We declared an algorithmic ceiling that was actually a compute/throughput
> ceiling, and pivoted away from the approach that won.**

Our `rl_research/EXPLORED_AND_ABANDONED.md` concluded that pure PPO had a
"structural ceiling" (0% vs Producer), that "capacity is not the bottleneck"
(a 98K-param agent matched a 2.6M one), and that learned value functions were
"too noisy" for this game. We then spent the competition on search surgery
(12 abandoned experiment clusters), behavior cloning, and ultimately forking
someone else's hand-written planner.

The winner ran the exact approach we abandoned — pure self-play PPO, no
imitation — and found that **every single time he scaled the model
(1M → 5M → 25M → 200M), performance jumped dramatically.** Our "structural
ceiling" was an artifact of running tiny models (≤2.6M params) for too few steps
on a slow Python simulator. Small-scale negative results do **not** generalize to
large scale. That is the Bitter Lesson, and it cut against us.

**Takeaway for Pokémon:** Do not conclude "RL can't do this" from experiments
that didn't have the throughput and model size to give RL a fair chance. Prove
the ceiling is algorithmic, not just under-resourced, before pivoting.

---

## Lesson 1 — Throughput is the master variable. Build for speed *first*.

The winner's entire result was unlocked by **rewriting the environment in Rust**
(~26K env-steps/sec, 6.3M steps/GPU-hour on B200), which is what made 15B steps
feasible. He preallocated pinned CPU↔GPU buffers, ran thousands of envs in
parallel, and did observation encoding inside the env.

We limped along on a Python simulator at ~1–6 games/sec. That scarcity is what
*forced* us into sample-efficient hacks (BC, search, planner-forking) — none of
which had a high ceiling. Our faster `fast_env` (~16×) arrived too late and was
still Python.

**Order of investment:** environment throughput → model scale → everything else.
Throughput is not a "nice to have later"; it determines which algorithms are even
viable. Measure real steps/sec on day one and treat raising it as a first-class
workstream.

## Lesson 2 — Self-play from scratch beats imitation. Don't cap yourself at a teacher.

We repeatedly tried to behavior-clone / distill the Producer planner and learned
the hard way: **"you cannot BC past your teacher."** Rich-representation BC
regressed to 18% despite **98.7% move-level accuracy**; DAgger only recovered to
~26%. Imitation anchors you to the teacher's ceiling, and our *entire final
submission was a fork of someone else's agent* — structurally capped at ~rank 140.

The winner used **pure self-play, zero imitation**, and blew past every
hand-crafted agent on the ladder. Self-play has no teacher ceiling; the opponent
is always exactly at your own skill level, so the curriculum is automatic.

**Takeaway for Pokémon:** Start from self-play, not from cloning the public
rule-based bots. Use strong heuristic agents as *evaluation opponents and a
warm-start sanity check*, not as the thing you imitate. (One nuance — a brief BC
warm-start to escape the random-flailing phase is fine, as long as you keep
training past it with RL and never treat the teacher as the target.)

## Lesson 3 — Keep the observation/action encoding low-level; let the model learn the dynamics.

We poured weeks into hand-engineered features: ~30 mission-scoring multipliers,
timeline simulation, "reinforce-risk" terms, search heuristics. The winner used
**raw entity tokens** (one per planet/fleet/comet) plus a handful of summary
tokens, and a minimal **target-based action head**. He *tried* the crutches we
would have reached for — action masks, cross-attention, discrete size bins — and
found them **unnecessary or actively harmful** (masking out "silly" moves made
the model *worse*, because it stopped internally modeling the physics).

His one real domain decision was tiny: let the model pick a *target planet*
instead of a raw launch *angle*. Everything else, the model learned.

**Takeaway for Pokémon:** Represent the observation as entities (Pokémon, cards,
energies, options) with minimal preprocessing. Don't encode "Crustle walls ex
damage" as a feature — let a big enough model learn it from outcomes. Save the
engineering budget for scale and throughput.

## Lesson 4 — A pointer/attention action head is the right fit for "pick from a list of legal options."

The winner's action head selected a target via a scaled dot-product
(attention-style) over candidate entity tokens — a *pointer* over a
variable-length set, with no fixed action enumeration and no need for an action
mask (the candidate set already encodes legality).

This is an *exceptionally* good fit for Pokémon TCG: the cabt engine hands us a
variable-length list of **only-legal** options every decision
(`obs.select.option`, 1 to ~1000+ entries). Encode each option as a token, encode
the board state as tokens, and use a pointer head to score options. We get
legality for free and never fight a fixed action space (which is exactly what
sank our early SB3 MultiDiscrete attempts in Orbit Wars).

## Lesson 5 — Don't anchor evaluation/training to a weak or wrong baseline.

We optimized against our own `apex` agent for **a month** before discovering it
was only median-tier and the public Producer crushed it ~100%. That month
optimized against the wrong opponent. Self-play sidesteps the anchor problem
entirely — your opponent tracks your own skill.

**Takeaway for Pokémon:** Self-play is the primary opponent. Maintain a small,
*honest* fixed evaluation suite (the strong public heuristic bots, mirror,
random) but never let a single weak baseline define "good."

## Lesson 6 — Measurement discipline is what makes the whole loop trustworthy.

The one thing we genuinely did well: rigorous arena evaluation — side-alternated
**paired seeds**, **n ≥ 120**, real environment. We measured a ~±4.5% A/A noise
floor on identical agents and learned to **never act on n < 100** (every
small-sample "win" regressed at high n). This is the only reason we didn't ship
dozens of phantom improvements.

**Takeaway for Pokémon:** Pokémon has *more* variance than Orbit Wars (shuffles,
coin flips, mulligans, hidden hands). Paired-seed, side-swapped, high-n evaluation
is mandatory, and the noise floor will be even larger. Budget for it.

## Lesson 7 — Stabilize self-play with best-checkpoint gating + a league.

The winner's PPO stayed stable via: a **frozen "last-best" checkpoint**
opponent, promoted only when the trainee beats it **>70%**; small **KL** and
**value cross-entropy** terms against that frozen checkpoint; advantage
normalization; entropy bonus; GAE-λ. Simple PPO + DDP scaled linearly across
GPUs — he chose it over IMPALA precisely for that simplicity.

His main *regret* was **not adding league play** (playing vs a pool of past
checkpoints). Pure self-play overfit to its own latest strategy and was fragile
to strategic cycles — especially in the non-transitive 4-player mode.

**Takeaway for Pokémon:** Use best-checkpoint gating from the start, and budget
for a **league / past-checkpoint pool** early. Pokémon strategy is
non-transitive (deck/line A beats B beats C beats A), so self-overfitting to one
opponent is a real risk even in a 2-player game.

## Lesson 8 — Reward & discount choices have non-obvious training-efficiency costs.

The winner kept **γ = 1.0** (so the win-probability critic stayed well-defined),
which produced a model that would build a lead and then *stall* — wasting
training compute on already-decided games. His fix-in-hindsight: early
truncation / surrender to keep states relevant.

**Takeaway for Pokémon:** Terminal win/loss = ±1 is the honest signal. If we add
shaping (e.g. prize differential), make it **potential-based** so it doesn't
change the optimal policy. Watch for degenerate compute-wasting equilibria
(stalling, deck-out farming) and add truncation if they appear.

## Lesson 9 — Plan for submission constraints (CPU, time budget, file size) as a real workstream.

To actually *ship* the 200M model the winner did serious engineering:
**int8 dynamic quantization** for CPU inference speed, **4-bit NormalFloat (NF4)
codebook quantization** (group size 128, fp16 scales) to fit the **100 MiB**
file cap, and a **5M-param fallback model** that took over when a slow CPU
threatened the per-turn time limit (the learned critic said most games were
already decided, and the small model converted 100% of winning positions).

**Takeaway for Pokémon:** Know our constraints *now* (per-turn time limit, bundle
size, CPU-only inference) and design the model so it can be shrunk/quantized to
fit. A model that wins in training but times out on Kaggle's CPU is worth zero.

## Lesson 10 — Agentic development works *when scaffolded* — but it's not a substitute for thinking.

The winner wrote ~no code himself (fully agentic via Codex) and it worked because
of **structure**: tight `CLAUDE.md`/`AGENTS.md` conventions, a single
`just prepare` gate (format + lint + type-check + test + **doc-freshness check**),
**parity fixtures** validating the Rust env bit-for-bit against real replays, and
a PR checklist. Mapped code changes *must* update mapped docs. This is what let
agents "rip" without producing a mess.

By contrast, our repo accumulated 30+ one-off scripts, 12 experiment clusters,
and heavy churn. The winner was explicit: coding agents were an *accelerator, not
a substitute for thinking* — his architecture, scaling strategy, and the decision
to bet on scale were human.

**Takeaway for Pokémon:** Set up the scaffolding (doc-driven workflow, a single
`prepare` gate, parity/eval tests) before generating lots of code. Let agents
implement specs fast; keep the strategy and the hard judgment calls human.

---

## What we did right and should keep

- **Rigorous paired-seed, high-n arena evaluation** (Lesson 6) — our best habit.
- **Extensive documentation of dead-ends** — the graveyard saved us from
  repeating mistakes. Keep `rl_research/` as the experiment log here too.
- **Expert Iteration loop mechanics** were sound (collect → search → distill);
  the failure was the *passive simulator opponent* and lack of scale, not the
  loop. (Pokémon is a POMDP with chance nodes, so deterministic search is harder
  here anyway — favors learned value over search.)
- **`uv` + `ruff` + `pyright` + Kaggle CLI tooling** — already mirrored here.

## How Pokémon TCG differs from Orbit Wars (so we don't over-copy)

| Dimension | Orbit Wars | Pokémon TCG (cabt) |
|---|---|---|
| Players / moves | 2p & 4p, simultaneous | 2p, **turn-based / alternating** |
| Information | near-perfect | **imperfect** (hidden hand, deck order, prizes) |
| Stochasticity | low | **high** (shuffle, coin flips, mulligan, draw) |
| Action space | structured (source/target/size) | **pick indices into a variable list of legal options** |
| Env | we could rewrite in Rust | **compiled C++ via ctypes; can't rewrite** — and it's a **global singleton** (`Battle.battle_ptr`), so parallelism is **process-level only** |
| Search | deterministic, tractable | chance + hidden info ⇒ search needs determinization; **favors learned policy/value** |
| Reward | win/loss ±1 | win/loss ±1 (+ optional prize-diff shaping) |
| Deck | fixed map | **deck is a fixed 60-card choice, not in the action space** — a separate meta-optimization, and a source of non-transitivity |

The biggest structural consequence: **we cannot Rust-rewrite the env**, so our
throughput lever (Lesson 1) is *massive process-level parallelism + efficient
batched policy inference*, not a faster single env. Measure the true ceiling
early — it determines everything downstream.

See [`SELFPLAY_RL_PLAN.md`](./SELFPLAY_RL_PLAN.md) for how these lessons turn
into a concrete plan.
