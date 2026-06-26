# Phase 1 — Representation research (2026-06-26)

Before committing to the Phase-2 model we paused to research how to encode the
observation and action space — the lever the Orbit Wars retrospective flags as
decisive for self-play RL (Lessons 1–4). Three sources: our own `~/git/orbit_wars`
repo, the academic record (AlphaStar / OpenAI Five / DouZero / Suphx / MTG), and —
most valuably — the **actual Kaggle kernels + discussion forums for this
competition**. This entry records what we found and what it changed.

## TL;DR — what changed in the plan

1. **Card-ID embedding is load-bearing — confirmed externally.** Keep it; it's the
   core of `encoding.py`.
2. **Inference limit is 600 s *per game*, not per turn** (1.6 vCPU / 8 GB / CPU-only).
   ⇒ latency is **not** the model-size ceiling; **RAM + bundle size are**. The
   scaling bet is *more* viable than P0.3 first implied. Fixed the P0.3 framing.
3. **The engine pre-sorts options best→worst; returning option [0] ("B1") beats
   random ~88–90%.** Our permutation-invariant pointer head was discarding that
   signal ⇒ added an **ablatable option-rank feature** (`opt_rank`) to the encoder.
4. **Evaluation is the real bottleneck, not the algorithm** (multiple teams).
   Vindicates building the eval harness first; sharpen the high-N discipline.
5. First model band = **`small`** (d256/L6, ~5.6M) for headroom (decided with the
   600 s/game ceiling in hand).

## Inference environment — the hard constraints (Kaggle #708810)

- **600 seconds TOTAL per game** — *no per-turn limit*. With ~136 decisions/game
  that's ~4.4 s/decision of headroom on average, spendable unevenly.
- **1.6 vCPU, 8 GB RAM, CPU-only** at submission. No GPU.
- Our P0.3 CPU numbers (`small` ~3–25 ms/decision, `medium` ~10–90 ms) ⇒ a *whole
  game* of `small` inference is well under 1 s; `medium` a few seconds; even a
  ~100–200 M model fits inside 600 s. **So model size at submission is capped by
  8 GB RAM + bundle/file size, not latency.** (Bundle cap: confirm in P6.1; Orbit
  Wars winner needed NF4 quant to fit 100 MiB — assume similar.)
- **Forward-search API (`search_begin/step`) is *not confirmed* to run in the
  Kaggle eval harness** (#713608, #708810). Our policy is feed-forward, so we don't
  depend on it; determinized MCTS stays a *training-time* tool only.
- **No RNG seed hook** (re-confirmed): engine takes only the 120 card IDs; every
  eval is an independent sample. Reinforces our unseedable finding (P0.4).

## Representation — what the field does, and our choice

**Card identity must be a learned embedding (the decisive finding).** Kaggle
#713608: **9 methods that encoded only card *type* + *damage* all plateaued at the
same "feature ceiling" (~68.5%)**; the first to break through was a **card-embedding
multi-head transformer (d=64, NHEAD=4, NLAYERS=2 → 73–74% vs mirror)**. "Card
identity is the critical feature." Our Orbit Wars retrospective says the same in
reverse (hand-engineered features were the trap, Lesson 3). The official RL/MCTS
sample kernel (kiyotah) also embeds raw card IDs. ⇒ **Our card-ID embedding over
1267 cards is the right bet, independently triple-confirmed.**

How others encode the observation:

- **kiyotah RL/MCTS sample** (this competition): a **sparse `EmbeddingBag`**
  (vocab 22 000, d=128) over **24 tokens** (8 bench + 2 active + 2 player-state +
  hand + deck + stadium + misc). Card ID = embedding index; value = presence or a
  scaled weight (energy 0.5, hand/deck 0.25). *Bag-of-cards per token.* Our
  per-entity tokens (one token per Pokémon/card, card-ID embedding + 15-d numeric
  features) are finer-grained — better for the pointer head's option→target links —
  at the cost of more tokens.
- **AlphaStar**: up to 512 entity tokens via a 3-layer self-attention transformer;
  scalar + spatial encoders; LSTM core.
- **OpenAI Five**: a big flattened engineered vector + 4096-unit LSTM.
- **DouZero** (DouDizhu): cards as 4×15 one-hot matrices; evaluates only legal
  moves (no mask). **Suphx** (Mahjong): hand-crafted 34×1 channels + CNN.
- **MTG** "generalised card representations" (2024): for an *evolving* pool,
  one-hot fails on unseen cards; features+text-embeddings+meta ensemble generalises
  (≈55% of human picks on unseen cards at scale). *Relevant later if the card pool
  grows mid-competition — a fallback if pure ID embeddings can't cold-start new
  cards.* For now the pool is fixed at 1267, so ID embeddings are fine.

## Action representation

- **Pointer / per-option scoring is universal here.** Every learning kernel scores
  the engine's pre-enumerated `select.option` list rather than constructing actions
  in a fixed space; AlphaStar/OpenAI Five select targets by attention over entity
  embeddings. Confirms our pointer head (Lesson 4).
- **The engine enumerates options best→worst** (#713608): **"B1" (return [0]) beats
  random ~88–90%** — a strong local optimum. A permutation-invariant pointer throws
  this away ⇒ we now feed the **option index as an ablatable positional feature**
  (`opt_rank`, model-side `use_option_rank` toggle). Gives a free prior + a fast
  self-play floor; ablatable so we can measure whether it actually helps.
- **Multi-pick** (`maxCount>1`): kiyotah **pre-enumerates ≤64 combinations** and
  scores each (simple, but explodes on big option sets — they hard-cap at 64). Our
  plan is an **autoregressive pointer** (pick, mask, re-score) — scales to any
  option-set size. AlphaStar factors multi-argument actions autoregressively;
  OpenAI Five uses independent heads. We keep autoregressive; validate vs the
  enumerate baseline in P2.3.
- **Masking**: AlphaStar/OpenAI Five/DouZero avoid training-time action masks
  (legality is encoded by the candidate set / hierarchical constraints). We get
  this for free — only legal options become tokens.

## Evaluation & measurement — the real bottleneck

Multiple teams converged on: **the wall is evaluator noise + measurement, not the
network or the algorithm** (#713608).

- A **privileged-critic probe** showed the value head *can* represent the signal
  (overfit-one-batch → corr 1.000; a 10-feature linear board model hits AUC 0.82,
  vs the NN head's 0.61 mid-training). "Outcome is not irreducibly noisy; hidden
  information is not the bottleneck." ⇒ when our value head looks weak, suspect
  data/eval before capacity.
- **"Short-eval inflation":** 77 % @200 games → 67 % @400. **Revalidate at N≥400.**
- **Hill-climbing below the noise floor:** a config "improved" 0.786→0.921 over
  rounds; re-run at 2 000 games = 1.37 pp of pure noise. Threshold was 0.5 pp.
- **Imitation ceilings below the teacher** (DAgger peak clone-acc 99 % ⇄ *worst*
  H2H 28 %; best H2H ~41 % vs 59 % teacher). Confirms Lesson 2.
- **Search value is inversely proportional to base-policy quality:** beam on a weak
  heuristic +11 pp; beam on a *strong* one −15 pp. ⇒ once self-play is strong,
  don't bolt crude search on top.
- **Survivorship bias in loss-only analysis:** "fixes" from reading 40 losses A/B'd
  to a −7.6 pp pooled regression. Always A/B, never ship from a loss post-mortem.

All of this is exactly why we stood up `eval_harness.py` first. Our measured **A/A
null = 50.0 % ±3.1 pp at n≈1000**, floor ~1.5 K games/arm for 5 pp, is the discipline
they wished they'd had. Possible upgrade: **SPRT-style sequential testing** to spend
fewer games per decision (future).

## Evaluation / ladder operations

- **Ladder rating is high-variance and front-loaded:** identical agents diverged
  150–400 rating points; "an agent's fate is largely decided by its first 5–10
  games" (#712621). ⇒ **trust local high-N eval, not a single ladder score**; submit
  twice. Two active submissions count; 5/day; ~24 games/day/sub.
- **One deck per submission** is an official ruling (#711741) — confirms "fix the
  deck" (we learn to *play*, deck choice is a separate pass).
- **Meta is fast-moving and Lucario-heavy** (~42 % of field; #709263, #713608).
  Our eval suite must grow to several archetypes (P5.1) so we don't overfit the
  mirror.

## Game-mechanics notes that touch encoding (#708586, #714030)

- Attacks whose effect can't fully resolve are simply **not offered** ⇒ we never
  have to encode "can this resolve."
- **Retreating/benching a Pokémon clears its effects** (e.g. a once-per-game attack
  lock resets) — a dynamic the model must learn from the board, not a static
  feature.
- **3 000-step cap → draw** on infinite loops; our eval `max_steps=4000` sits above
  it, so the engine resolves the draw itself.

## Sources

- Kaggle threads: #708810 (inference env), #713608 ("What We Tried, What
  Ceilinged"), #708586 (sim vs official rules), #712621 (scoring variance), #711741
  (one deck/sub), #709263 (meta notes), #714030 (step-limit/loops), #714189
  (format).
- Kaggle kernels (cached `outputs/kernels/`): `kiyotah__reinforcement-learning-and-mcts-sample-code`,
  `makimakiai__ptcg-tiny-rl-to-submission-baseline-guide`,
  `kokinnwakashuu__ptcg-public-915-lucario-search-baseline`, et al.
- `~/git/orbit_wars`: `rl_research/EXPLORED_AND_ABANDONED.md` (12 closed clusters),
  `v2/model.py` (pairwise source→target + factored-fraction action head),
  `LEADERBOARD_CLIMB_PLAN.md` (rich-BC ceiling).
- Academic: AlphaStar (pointer/autoregressive heads), OpenAI Five (independent
  heads), DouZero (PMLR v139), Suphx (arXiv 2003.13590), MTG generalised card
  representations (arXiv 2407.05879).
