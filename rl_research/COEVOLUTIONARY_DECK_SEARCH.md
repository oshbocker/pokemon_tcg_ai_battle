# Coevolutionary Deck Search — Genetic Optimization Meets Game Theory

**Status: design only, not yet built. Speculative.** This is the dated design/
experiment log for the *deck meta-optimization* thread: searching deck space by
treating a trained agent as the fitness function for a deck. It is deliberately
kept **out of the Strategy writeup** until validated. If it works, it graduates
into the writeup; if it's a dead-end, it earns at most a short "we tried X, here's
why it didn't pan out" paragraph there. Record dead-ends and rejected designs
here too.

Companion docs: [`STRATEGY_WRITEUP_LOG.md`](STRATEGY_WRITEUP_LOG.md) (brief
pointer entry), [`LESSONS_FROM_ORBIT_WARS.md`](LESSONS_FROM_ORBIT_WARS.md)
(throughput/scale/self-play thesis), `docs/rl-obs-action.md` (encoding contract).

---

## 1. Premise & the architectural enabler

Above "train a strong agent for a fixed deck" sits a second optimization:
**deck selection**. Framing — a trained agent *is* the fitness function for a
deck, so deck search = "adapt an agent to this deck, then measure it."

**Why this is cheap: deck identity is not a model input.** The only place cards
enter the net is the learned **card-ID embedding** (`CARD_VOCAB=1268`, one row per
card) plus the trunk weights, which during training absorb the fixed deck's
draw/combo/sequencing structure. The 60-card decklist is returned only at
`select is None` and is otherwise outside the observation (`global_feat` carries
`deckCount`, a scalar, never contents). **Swapping a card changes no tensor
shapes** — no re-architecting, no re-init. Two regimes follow:

- **Zero-shot swap** (no retrain): fine for well-trained, functionally-similar
  card IDs; degrades sharply for novel cards whose embedding row is ~untrained.
  → a near-free **coarse pre-filter**, not a verdict.
- **Warm-start fine-tune** (load parent `best.pt`, keep 1266/1267 embedding rows +
  trunk + value head, adapt only the new card + sequencing): should converge far
  faster than a cold run. **This is the enabler** — affordable per-mutation
  evaluation is what makes a population search viable.

---

## 2. Implementation pieces (land in this order)

### Piece 1 — static card-metadata feature (`encoding.py` + `model.py`)

Give a *novel* card a sensible representation even when its learned embedding row
is cold, by adding a **frozen** projected feature keyed by card ID. Docs §0
deliberately omitted this ("let the ID embedding learn from outcomes"); the bet is
that a frozen metadata table (a) makes zero-shot swaps a real signal and (b)
shortens warm-start fine-tuning. Source: `CardData` from `all_card_data()` exposes
`cardType, hp, retreatCost, weakness, resistance, energyType, basic/stage1/stage2/
ex/megaEx/tera/aceSpec, evolvesFrom, attacks` — plus attack `damage`/`energies`
via `all_attack()`.

- `scripts/build_card_metadata.py` → committed `src/ptcg_battle/card_metadata.npy`
  `[CARD_VOCAB, META_DIM]`, normalized ~[0,1], row 0 = PAD zeros.
- `encoding.py`: add `META_DIM` + torch-free cached `load_card_metadata()`.
- `model.py`: `ModelConfig.use_card_meta` **ablation flag** (mirrors
  `use_option_rank`); frozen `register_buffer("card_meta", …)` + learned
  `card_meta_proj = nn.Linear(META_DIM, d)`; add its projection to entity and
  option tokens wherever `card_emb(...)` is added. As a buffer it rides in
  `state_dict()`; old checkpoints load into `use_card_meta=False` with no missing
  keys.
- Tests: shape/finite/range/row-0 + a build-vs-live-engine sync check (spirit of
  `test_declared_vocab_bounds_live_engine`).
- **Gate:** A/B `use_card_meta` on a *fixed* deck; must be ≥ neutral before we
  lean on it for transfer. This ablation is itself writeup material.

### Piece 2 — warm-start fine-tune (`--init-ckpt` on `train_selfplay.py`)

One flag; a warm start *is* the existing loop with a different init + shorter
schedule. Load parent weights `strict=False` (tolerate a `use_card_meta`
mismatch); seed `frozen_best`/pool from the parent when `--gate`. Recipe vs cold:
~⅓–¼ the iters, lower LR (`1e-4 → 3e-5`), same league/manifest + `--gate-vs pool`
so fitness is measured against the real opponent set. Optional
`--warmup-freeze-trunk N`: train only `card_emb + card_meta_proj + heads` for the
first N iters so the new card's embedding catches up before the policy shifts —
directly counters the "short fine-tune underrates good cards" bias.

### Piece 3 — coevolutionary mutation search (`scripts/deck_search.py`)

The orchestrator. See §3 for the population/fitness design. Two-stage funnel:
zero-shot pre-filter (Piece 1 makes it meaningful) → warm-start fine-tune
survivors (Piece 2) → high-n league eval → select → persist. Fully resumable from
`population.json`. Reuse, don't rebuild: fitness eval = existing `eval_harness`/
`scripts/eval.py`; legality checks = Piece 1 metadata; opponent weighting =
existing PFSP.

---

## 3. Population design: objective, axis-free diversity

### Decision: coevolution, NOT MAP-Elites (bitter-lesson rationale)

**Rejected: MAP-Elites / quality-diversity.** It requires hand-picking behavior
descriptor axes (aggression, curve, composition, …). That injects a *designer's
bias* about which strategic dimensions matter into the agents we select for —
against Sutton's bitter lesson. Recorded as a considered-and-rejected design.

**Chosen: competitive coevolution.** Let the **population itself be the axis**.
Fitness = performance against the other decks; a deck's "behavior" is its
**outcome vector against the population + archive** — emergent and learned, never
declared. This is the self-play-league lineage (AlphaStar League, PFSP,
fictitious self-play), which we already have machinery for.

### Fitness = success against the population (competitive fitness sharing)

Not raw win-rate. Use **competitive fitness sharing** (Rosin & Belew): weight each
defeated opponent `j` by `1 / (# population members that also beat j)`. Beating a
widely-beaten opponent earns little; being the *only* answer to some opponent
earns a lot. This **objectively rewards covering a distinct niche** — the niche
being "opponents I uniquely handle," discovered from outcomes, not a designer
grid. It is the axis-free analog of a MAP-Elites cell.

Diversity is then preserved by **non-transitivity** (deck matchups are RPS-like;
"beat the population" cannot collapse to one deck because none dominates) + fitness
sharing + the archive (below). Reuse **PFSP `var` weighting `4p(1-p)`** to
concentrate fitness games on *contested* matchups (fixes coevolutionary
*disengagement*, where a saturated matchup yields no gradient).

### Hall of Fame + resurrection (anti-forgetting)

Naive coevolution **cycles / forgets** (Red Queen in place: chase current meta,
forget old counters, A→B→C→A forever while looking like progress). Counters:

- **Hall of Fame** — past champions become *permanent opponents* in the fitness
  eval; a new deck must beat the history, not just the current pop. Machinery
  exists: never-evicted pool entries + `--league-checkpoint` + PFSP.
- **Resurrection** — periodically re-inject an archived champion (or lineage) into
  the *breeding* pool so a fallen genotype gets a fresh shot at the evolved meta.
  Reservoir-sampled HoF if unbounded gets too costly.

### "Remember the fittest" = Nash averaging (not a scalar champion)

Maintain the empirical payoff matrix `M[i][j]` = agent-on-deck-i vs
agent-on-deck-j win-rate over elites+HoF, and take the **Nash equilibrium of that
meta-game**. Its *support* is an emergent, non-redundant, **diverse** set chosen by
an objective (game-theoretic) criterion with zero designer axes. Gives us:
learned diversity/novelty (does a deck shift the Nash support?), the rubric's
robustness (a max-min/Nash-mixture deck is by construction not matchup-reliant),
and the **A/B hedge derived rather than guessed** (Nash support → the two decks
that best cover each other, replacing hand-picked Archaludon/Alakazam). Cost: the
matrix is O(N²) fine-tuned matchups → keep N small, PFSP-sample, reuse
`eval_harness`; compute Nash *periodically / at end*, run PFSP-weighted
fitness during the search.

### The residual designer bias (unavoidable, benign)

The one remaining hand-authored choice is the **mutation operator** (what a legal
swap is) — a *rules* constraint (60 cards, ≤4 copies except basic energy, ≤1 ACE
SPEC, keep ≥1 basic Pokémon; all checkable from Piece-1 metadata), not a value
judgment about which strategies are good. That's the bias floor.

### Selection & the generational loop

- Individual = `(deck, checkpoint, fitness-record)`; genotype = deck; phenotype
  needs the fine-tuned checkpoint. `deck_hash` = sorted card multiset → dedup key
  (never re-evaluate a known deck; cache its fitness).
- (μ+λ) with **soft** elitism: elites (top by *shared* fitness) breed and carry
  forward; λ offspring by mutation/crossover of elites (rank/tournament parent
  choice — win-rates sit in a narrow band, so raw fitness-proportional barely
  discriminates).
- Offspring generation split ≈ **65% local mutation / 20% cross-archetype
  crossover** (multiset blend of two elites) **/ 15% novelty** (macro-mutation of a
  whole evolution line/package, or restart from a meta decklist) — local mutation
  alone only explores a champion's neighborhood; novelty jumps archetypes.
- Persist `population.json` + `hall_of_fame/` + append-only `lineage.csv`
  (parent, swap, zero-shot score, post-FT fitness, CI, league_version). Resumable.

### Noise & non-stationarity guards

- Fitness comparisons use the **Wilson lower bound** (`eval_harness`), so an
  under-sampled lucky deck can't dethrone a well-sampled one.
- **Elite/incumbent re-evaluation each generation** (accumulate `n` → tighter CI);
  flukes regress out and get demoted.
- Stamp every fitness with a **`league_version`**; when the opponent set advances,
  **re-score elites against the new set before selecting** (else you compare fresh
  offspring to stale elites — the Red Queen trap). Safe first mode: **freeze the
  league during a search campaign**, advance it between campaigns; co-evolving the
  league concurrently is v2.

---

## 4. Validation ladder (de-risk before scaling)

1. **Kill-criterion experiment (do first, ~1 afternoon):** take the current
   `best.pt`, zero-shot-eval it on a 1-card-swapped deck vs the manifest, and
   separately warm-start fine-tune ~20–30 iters. If warm-start doesn't recover to
   ~parent strength *dramatically* faster than a cold run, the whole premise is
   suspect — stop and rethink.
2. **Piece 1 ablation:** `use_card_meta` ≥ neutral on a fixed deck.
3. **Piece 2 single hand-picked swap:** confirm warm-start reaches parent strength
   in far fewer iters than cold (measures the enabler before automating).
4. **Piece 3 small run:** μ/λ ≈ 6/18, frozen league, a few generations; check the
   diversity monitors (below) actually hold and the champion improves on a
   held-out opponent set.

Diversity monitoring per generation (trip a signal / auto-inject novelty on
collapse): # distinct lineages surviving, mean pairwise deck distance
(`½·|symmetric difference|`), card-usage entropy, Nash-support size.

---

## 5. Open decisions

- **Fitness compute:** cheap PFSP-weighted win-rate-vs-population each gen, vs the
  fuller O(N²) payoff matrix + Nash. Leaning: PFSP-weighted during search, Nash
  only periodically / at end to read off the portfolio.
- **Archive policy:** unbounded HoF vs reservoir-sampled; resurrection cadence.
- **μ/λ** (per-generation compute budget — each offspring = one warm-start
  fine-tune) and **elite re-eval cadence** (every gen safest; every-other if tight).
- Whether to run the search from *both* A (Archaludon) and B (Alakazam) parents —
  a swap that helps one archetype may not transfer.

## 6. Log

### 2026-07-01 — design captured
Split out of the STRATEGY_WRITEUP_LOG discussion. MAP-Elites considered and
rejected (designer-axis bias, bitter lesson); pivoted to competitive coevolution +
competitive fitness sharing + Hall of Fame/resurrection + Nash-averaging for the
axis-free "keep the fittest, preserve diversity" objective. Nothing built yet;
next action = the §4.1 kill-criterion experiment.

<!-- Append new dated entries above this line. -->
