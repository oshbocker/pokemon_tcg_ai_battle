# Strategy Writeup Log

**Living document.** Append dated notes here as our approach evolves. At the end,
this feeds the Strategy Category writeup — the competition with the real prize
money. Record *why* we made each decision, what hypotheses we tested, and what
we learned (including dead-ends). The rubric rewards depth of reasoning, not just
leaderboard rank.

## Why this doc exists (the Strategy Category)

The **Strategy Category** (`pokemon-tcg-ai-battle-challenge-strategy`) is a
separate competition from the Simulation ladder and holds the prize money:

- **$240,000 total** — eight (8) finalists get **$30,000 each**; finalists may
  be invited to an in-person tournament in Tokyo (TBD).
- Entering Strategy **requires** also competing in Simulation on an identical
  team (cross-division team must match exactly).
- Deliverable: a **single Kaggle Writeup** (it's a hackathon — **one submission
  only per team**).
  - **≤ 2000 words** (over-limit → penalty). Title + subtitle + detailed
    analysis. Must select a Track.
  - Optional Media Gallery (images/videos). Images that violate the Pokémon
    Elements license → **disqualification**.
  - **Draft/un-submitted writeups at the deadline are NOT considered.**
  - Private Kaggle resources attached to the writeup become **public after the
    deadline**.

### Judging rubric — write to this

| Weight | Category | What they score |
|---|---|---|
| **70%** | **Model Score** | clarity of approach & rationale; originality + technical soundness; **consistency across repeated matches under stable conditions**; **avoiding over-reliance on specific initial states, matchups, or situational advantages**; performance within the track |
| **20%** | **Deck Score** | deck concept clarity + alignment with intended strategy; key-card selection & utilization |
| **10%** | **Report Score** | logical structure & writing; effective figures/charts/tables |

Key framing from the organizers: *"High leaderboard ranking may provide an
advantage in performance scoring, but it does not guarantee a strong result in
the Strategy Category. Participants in middle or lower tiers can still achieve
high overall scores through deep analysis, originality, and well-structured
reporting."*

**Robustness is called out twice** (consistency across repeated matches;
low reliance on luck/matchups) — this maps directly to our league / exploiter /
PFSP self-play work. Make that story explicit in the writeup.

## Timeline (both categories)

- **Jun 16, 2026** — both categories start.
- **Aug 9, 2026** — Simulation entry + team-merger deadline.
- **Aug 16, 2026** — Simulation final submission deadline.
- **Aug 17 → ~Aug 31, 2026** — Simulation leaderboard keeps running to
  convergence, then final.
- **Sept 13, 2026** — **Strategy writeup final submission deadline.**
- **Sept 14 – Oct 11, 2026** — Strategy judging period (subject to change).
- **TBD** — results announcement.

### Submission timing

No scoring advantage to submitting the writeup early, and it's one-shot — so
submit right before **Sept 13**. But the *content* should reflect the final
agent + final leaderboard, which don't settle until the Simulation deadline
(**Aug 16**) and convergence (**~Aug 31**). Plan: keep appending notes here
throughout → lock the narrative once the sim leaderboard is final → submit near
Sept 13.

## Writeup outline (draft — evolve as we go)

Aim the structure at the 70/20/10 rubric:

1. **Approach & rationale** (Model, 70%) — why self-play PPO over feature
   engineering / imitation (see [LESSONS_FROM_ORBIT_WARS.md]); throughput-first
   thesis; architecture, encoding, training loop.
2. **Robustness story** (Model, 70%) — league + exploiters + PFSP; how we avoid
   luck/matchup over-reliance; eval-harness Wilson-CI evidence of consistency.
3. **Deck construction** (Deck, 20%) — deck-selection logic, key cards, how the
   deck aligns with the agent's game plan (see [DECK_SELECTION_STRATEGY.md],
   meta snapshots).
4. **Results & figures** (Report, 10%) — charts: training curves, matchup
   matrices, ablations (option-rank, gate calibration).

## Dated strategy notes

### 2026-07-01 — doc created
Set up this log after reading the Strategy Category requirements. Current agent
state: RL self-play with a deck-agnostic league; Archaludon (A) + Alakazam (B)
agents; PFSP wired into training. Known weaknesses being worked (see
[RL_LADDER_ANALYSIS_2026-06-30.md]): first RL sub loses to Alakazam combo and
Starmie tempo decks not yet in the pool → widening the pool. These matchup gaps
are exactly the "over-reliance on specific matchups" risk the rubric penalizes,
so fixing them is both a ladder and a writeup win.

### 2026-07-01 — deck meta-optimization: warm-start fine-tune + mutation search
The next optimization layer above "train a strong agent for a fixed deck" is
**deck selection itself**. Framing: a trained agent *is* the fitness function for
a deck, so deck selection becomes a search where each evaluation is
"adapt an agent to this deck, then measure it."

**Key architectural fact that makes this cheap.** Deck identity is *not* an input
to the net. The only place cards enter the model is the learned **card-ID
embedding** (`CARD_VOCAB=1268`, one row/card) plus the trunk weights, which during
training absorb the fixed deck's draw/combo/sequencing structure. The 60-card
decklist is returned only at `select is None` and is otherwise outside the
observation (`global_feat` carries `deckCount`, a scalar, never contents). So
**swapping a card changes no tensor shapes** — no re-architecting, no re-init.

Consequences we're betting on:
- *Zero-shot swap* (no retrain): works for cards whose ID embedding was
  well-trained and functionally similar; degrades sharply for novel cards whose
  embedding row is ~untrained. Useful as a **near-free coarse pre-filter**, not a
  verdict.
- *Warm-start fine-tune* (load parent `best.pt`, keep 1266/1267 embedding rows +
  trunk + value head, adapt only the new card + sequencing): should converge far
  faster than a cold run. **This is the enabler** — it makes per-mutation
  evaluation affordable, which makes a genetic/hill-climb deck search viable.

**Highest-leverage change identified:** add **static card metadata** (type, HP,
retreat, weakness/resistance, stage, ex/mega/ace flags — all in `CardData` from
`all_card_data()`) as a *frozen* projected feature keyed by card ID. Docs §0
deliberately left this out ("let the ID embedding learn it from outcomes"). But a
frozen metadata table gives a *novel* card a sensible representation even when its
learned embedding row is cold → (a) makes zero-shot swaps a real signal and
(b) shortens warm-start fine-tuning (the model isn't relearning "what is this
card" from scratch). This directly multiplies how many mutations we can afford.
Make it an **ablation flag** (`use_card_meta`, mirroring `use_option_rank`) so we
can A/B whether it helps — that ablation is itself writeup material.

**Robustness caveats (which are also the writeup's 70% story):** fitness is noisy
(unseeded engine → high-n side-swapped eval, Wilson CIs), short fine-tunes bias
*against* genuinely-good cards (embedding hasn't caught up → underrating), and
deck fitness is **non-stationary** — it depends on the opponent pool, so search
must evaluate against the *league/meta*, not a strawman, or we optimize a mirage.
This ties straight to the rubric's "avoid over-reliance on specific matchups."

Full design lives in its own doc — [`COEVOLUTIONARY_DECK_SEARCH.md`](COEVOLUTIONARY_DECK_SEARCH.md)
(genetic optimization meets game theory: warm-start fine-tune + competitive
coevolution, competitive fitness sharing, Hall-of-Fame/resurrection, Nash-averaging
for axis-free diversity). Kept **out of this writeup log in depth on purpose**:
it's speculative and unproven. It graduates into the writeup only if it works;
if it's a dead-end it earns at most a short "tried X, here's why not" paragraph.
Status: **design only, not yet built.**

### 2026-07-01 — replay audit of both live submissions (Archaludon A / Alakazam B)
Pulled 35 recent ranked replays for each of our two live submissions and
attributed win/loss by matching our exact 60-card fingerprint (the seat whose
decklist is constant across all of a submission's games is ours). Tooling made
reusable as the `scout-agents` skill (`scripts/scout_agents.py`). Both agents
landed at **16‑19 (46% decisive WR)** against the live field in this window,
consistent with their ~1047 / ~913 skill ratings sitting around/below the ladder
median — neither is a clear ladder winner yet.

**Archaludon (A, LB 1047)** — *strength:* wins the fair ex-vs-ex slugfests and
wins them faster (avg 112 steps in wins) — 4‑0 vs other Archaludon variants,
2‑0 Cynthia's Garchomp, 2‑1 Mega Starmie, clean singles vs Blissey / Ogerpon /
Seaking / Iono's Bellibolt / Mega Abomasnow. *Killer weakness:* **Alakazam combo
2‑11 (15%)** — 13 of 35 games were vs Alakazam (the ladder's #1 deck, ~32%
[[meta-snapshot-2026-06-30]]), so it runs into its worst matchup constantly,
which caps its rating. Also 0‑3 Marnie's Grimmsnarl. Losses run long (143 vs 112
steps) → ground out / OHKO'd once the opponent's engine comes online; 10/19
blowouts but 4 lost by a single prize (winnable with better late-game lines).

**Alakazam (B, LB 913)** — *strength:* **beats Archaludon 7‑2 (78%)**, exactly
the anti-Archaludon hedge it was added for; also 3‑2 Mega Lucario, 2‑1 Mega
Starmie. *Weaknesses:* **Hop's Snorlax 0‑4** (stall/deckout out-grinds the
combo), **Dragapult 2‑5 (29%)** (spread/bench damage disrupts setup), Alakazam
mirror 1‑3. Loss shape is the tell: losses are very long (168 steps avg, several
240‑261) and **13/19 are blowouts where it took only 0‑1 prizes** — the combo
never comes online under disruption. This is a **training-distribution gap**
(few spread/stall decks in its pool), not late-game value tuning.

**Two strategic reads (both map to the rubric's anti-matchup-reliance 70%):**
1. Our two agents are a genuine **rock‑paper‑scissors** — Alakazam > Archaludon
   (78%) — yet Archaludon is the *higher-rated* flagship. The hedge-pair logic
   holds, but Archaludon's Alakazam matchup (13/35 games, 15% WR vs the #1 meta
   deck) is the single biggest rating lever: add **Alakazam exploiters** to
   Archaludon's self-play pool, or reconsider it as flagship.
2. Alakazam's blowouts vs Dragapult/Snorlax are a **setup-fragility /
   pool-coverage** problem, not a close-game one — add spread + stall/deckout
   decks to its opponent distribution.

Caveat: n=35/agent, one time window; `(MIRROR)` rows are self-play/identical-deck
opponents and low-signal. Re-run `scout-agents` after pool changes to confirm the
Alakazam and Snorlax/Dragapult gaps close.

### 2026-07-01 — act on the audit: Alakazam-harden Archaludon (warm-start resume)
Direct response to the replay audit above (Archaludon **2‑11 / 15%** vs Alakazam
combo, the ladder's #1 deck). Root cause: the shipped Archaludon *did* train
against `kaggle:alakazam` @1.5 with PFSP, but that hand-coded rule agent is far
weaker than the real Alakazam pilots it meets on the ladder (strong humans + our
own RL Alakazam), so training parity there didn't transfer. Fix = **harder
Alakazam sparring**, resumed from the existing best rather than retrained cold:

- **New trainer manifest** `agent/opponents/mixed_pool_archaludon_v2_alakazam.json`
  — bumps the heuristic `kaggle:alakazam` exploiter 1.5→2.0; keeps `kaggle:archaludon`
  held out as the honest eval yardstick.
- **Strong exploiter added:** the trained **Alakazam RL best.pt** (our LB‑913 B agent,
  `selfplay_alakazam_medium/best.pt`) enters the league via
  `--league-checkpoint …:2.0:alakazam_deck.csv:alakazam_rl`, piloting its own deck.
  Combined Alakazam-archetype pressure (~4.0) is the dominant fixed bucket, by design.
- **Warm-start resume:** `--init-ckpt probe_archaludon_medium_long/best.pt` (it850)
  seeds the trainee + the gate's frozen_best + a never-evicted `pool['parent']` anchor
  (can't silently regress below LB‑1047). LR 5e‑5→5e‑6, 500 iters, PFSP(var) after 50 —
  re-adapt to the harder pool, don't overwrite a competent net.
- Launch cell added to `notebooks/colab_selfplay_archaludon.ipynb` (runs on Colab/L4;
  no local GPU). Wiring smoke-tested locally on CPU: warm-start `state_dict` match is
  exact, and both `model:alakazam_rl` and `parent` appear in the realized opponent mix +
  gate breakdown.

**Hypothesis under test (writeup-relevant, the 70% anti-matchup-reliance criterion):**
exposing the agent to a *strong, learned* copy of its worst matchup — not just the
matched heuristic — is what closes a ladder blind spot. **Success = eval
`kaggle:alakazam` + the gate `alakazam_rl` bucket climb out of the teens toward parity
without starving starmie/dragapult**; confirm by re-running the `scout-agents` skill on
the ladder after the next submission. Watch for regression on the held-out
`kaggle:archaludon` yardstick (over-indexing on Alakazam). Known un-addressed gap:
Marnie's Grimmsnarl (0‑3 on ladder) is still absent from the pool.

### 2026-07-01 — competitive meta research → coevolution deck seeds (real vs Kaggle)
Ran a deep-research sweep (r/pkmntcg + LimitlessTCG/RK9/NAIC, adversarially
verified) and reconciled it against our Kaggle ladder meta + card pool. Full
digest: [`META_RESEARCH_2026-07-01.md`](META_RESEARCH_2026-07-01.md). Two findings
reshape the coevolution seeding plan:

**(1) Our card pool is CURATED — real decklists don't copy 1:1.** Every meta
archetype's Pokémon lines are legal, but universal Standard staple *trainers* are
ABSENT (Iono, Arven, Professor's Research, Nest Ball, Counter Catcher, Super Rod;
special Darkness Energy). The engine's draw/disruption shell is a different set
(Dawn/Hilda/Cheren/Judge/Boss's Orders/Ultra Ball). So an archetype's Pokémon core
transfers but the **trainer shell must be *discovered*, not copied** — which is
exactly what makes the coevolution search valuable (the optimal in-pool 60 is
unknown). Bonus reconciliations: our **Alakazam is likely stronger in Kaggle than
real** (its real predator *Iono* isn't in-pool); **Archaludon's** real-world
decline may not transfer, but see (2).

**(2) Real-world tier = archetype *headroom*; Kaggle meta = current fitness.**
Verified standings vs our measured Kaggle numbers:
- **Alakazam (B):** real tier-1.5 (NAIC 2026 2nd) AND Kaggle #1 (32%/51%) → keep, top seed.
- **Archaludon (A):** Kaggle #2 (13%/49%, our best agent) but real-world **declined /
  not tier-1-2** → keep (ladder rewards Kaggle strength) but treat as **lower-ceiling**;
  let stronger seeds overtake it in coevolution rather than over-investing.
- **Marnie's Grimmsnarl ex:** strong in BOTH (real tier-1; Kaggle 7%/57%, the deck
  that beat our Archaludon) — closes the earlier "is Grimmsnarl worth it" question: yes.
- **Dragapult ex:** real format-defining (~4/8 NAIC top-8) but only 4% of Kaggle pilots
  → likely **Kaggle-underexploited alpha**.

**Roster decision for coevolution seeds:** add **Grimmsnarl** (spread/disruption
control) and **Dragapult** (tempo/bench-spread) as high-confidence, strategically-
distinct, in-pool-legal additions to the Metal-control + Psychic-combo pair; a 3rd
seed is a fork (Clefairy single-prize aggro [real NAIC winner] / Chandelure [Kaggle
60%] / Crustle stall [orthogonal, hardens vs the stall decks our agents fold to]).
This gives coevolution a diverse, from-strength starting population. The real-vs-Kaggle
*divergence itself* (Dragapult underplayed here) is the kind of exploitable gap the
search should find. Writeup-relevant: this is the "deck concept + alignment" 20% and
feeds the robustness (matchup-diversity) 70% story. Builds on [[coevolutionary-deck-search]].

Un-verified / to pull fresh before seeding: exact Dragapult & Clefairy counts (only
Alakazam + Grimmsnarl lists were verified); several meta-share numbers were
adversarially refuted (see digest caveats).

### 2026-07-01 — coevolution kill-criterion: warm-start fine-tune PASSES (vs cold)
The gating experiment for the whole coevolution deck-search (design:
[`COEVOLUTIONARY_DECK_SEARCH.md`](COEVOLUTIONARY_DECK_SEARCH.md) §4.1): can a
*short* warm-start fine-tune give a usable per-mutation deck-fitness signal? If
warm ≈ cold at an affordable budget, the "trained-agent-as-deck-fitness" idea is
dead. Setup: deck = `archaludon_judge_swap` (1-card swap from the parent's deck);
both runs 30 iters / 48 games-iter / CPU / same mixed_pool league + pool gate
(>55% floor). WARM = `--init-ckpt probe_archaludon_medium_long/best.pt`, LR 3e-5→1e-5.
COLD = scratch, LR 3e-4.

**Result — decisive:**
| | WARM (init from parent) | COLD (scratch) |
|---|---|---|
| entropy | healthy ~0.82 throughout, no collapse | **collapsed to 0.000 by it3** (KL breaker tripped it1) |
| gate-pool @it10 | **63.4%** (PROMOTED) | **18.6%** (never promoted → no `cold/best.pt`) |
| gate-pool final | **68.5%** @it30, monotonic ↑ | still collapsed; killed at it12 |
| per-opp @best | archaludon 77 / starmie 83 / dragapult 73 / alakazam 60 | archaludon 7 / starmie 0 / dragapult 0 / alakazam 13 |

Warm re-adapts to the swapped deck in ~10 iters (~7-20 min on L4); cold self-play
from random init at that budget collapses (the exact failure the 1200-iter
anti-collapse recipe exists to avoid). **Verdict: warm-start-as-fitness is
validated for SMALL mutations → the coevolution search is viable.** Caveat that
bounds the claim: a 1-card swap is the *easiest* case (nearly all card-ID embedding
rows already trained); the edge shrinks for mutations adding NOVEL cards (cold
embedding rows) — the design's frozen card-metadata feature is the proposed fix and
the natural next experiment. Stopped the redundant cold run (collapsed, foregone).

**Sharper test running:** high-n side-swapped `ref` (parent on original deck) /
`zero-shot` (parent on swapped, no fine-tune) / `warm` eval (160 games × 7
mixed_pool opponents, Wilson CIs) → `outputs/coevo_kill/eval_{ref,zeroshot,warm}.csv`.
The decision-relevant number is **warm vs zero-shot**.

**Results (160 games/opp side-swapped, Wilson 95% CI; unweighted mean across 7):**
| Opponent | ref (orig deck) | zero-shot (swap, no FT) | warm (swap, FT) | warm−zeroshot |
|---|---|---|---|---|
| kaggle:alakazam | 54.4% | 55.0% | **66.9%** | **+11.9** |
| kaggle:archaludon | 67.5% | 65.0% | 70.0% | +5.0 |
| kaggle:starmie | 91.9% | 88.8% | 88.1% | −0.7 |
| kaggle:dragapult | 73.8% | 71.9% | 69.4% | −2.5 |
| kaggle:romanrozen | 88.8% | 81.2% | 80.0% | −1.2 |
| heuristic | 86.2% | 84.4% | 85.0% | +0.6 |
| random | 100% | 100% | 100% | 0 |
| **mean** | **80.4%** | **78.0%** | **79.9%** | **+1.9** |

**Interpretation — the useful finding is NOT the aggregate.** On a 1-card swap,
zero-shot already sits ~ref (78.0 vs 80.4) → it's a fine near-free *coarse* filter.
Warm-start recovers to ref (79.9) — but the aggregate +1.9 hides that the gain is
**concentrated in one matchup: Alakazam, +11.9pp** (zero-shot 55→warm 67, z≈2.2,
p≈0.03), while every other matchup is warm≈zero-shot (noise). Why: the swapped card
is **Judge** (shuffle-draw-4, hand disruption), a *targeted anti-Alakazam tech* —
Alakazam's combo damage scales with hand size. **The mutation's value is LATENT: the
parent already had Judge in-deck (zero-shot) but hadn't learned to *play* it vs
Alakazam; only fine-tuning realizes the +12pp.** This is the core argument for
warm-start-as-fitness over zero-shot: **zero-shot systematically undervalues cards
whose payoff needs learned sequencing** — exactly the matchup-fixing mutations a deck
search exists to find. (Bonus: this is a concrete, real anti-Alakazam deck fix,
complementary to the RL-exploiter pool run — the search would find it automatically.)

**Conclusions for the plan:**
1. Kill-criterion PASSED — warm-start is a valid, cheap per-mutation fitness (recovers
   ref where cold collapses).
2. Adopt the **two-stage fitness protocol** (now evidence-backed): zero-shot coarse
   pre-filter to cull cheaply → warm-start fine-tune on survivors to realize+confirm
   latent matchup gains that zero-shot cannot see.
3. **Next experiment = the real bottleneck:** all of the above is the *easy* 1-card
   swap (embedding rows already trained). NOVEL-card mutations will crater zero-shot
   and stress warm-start. Test the **`use_card_meta` frozen-metadata feature** as the
   fix: does it let warm-start recover strength on a mutation that adds a card the
   parent never saw? That gates whether the search scales past tiny swaps.

### 2026-07-01 — novel-card kill-criterion: frozen card-metadata (`use_card_meta`) PASSES

The follow-up the §4.1 entry above flagged as "the real bottleneck": a 1-card
swap of an *already-trained* card is the easy case — does warm-start-as-fitness
survive a mutation that introduces a card the parent **never saw** (cold
embedding row)? Proposed fix, now built: **`use_card_meta`** — a frozen
`[1268, 53]` static card-metadata table (`src/ptcg_battle/card_meta.py`:
category/stage/ex-mega-ACE flags, HP scalar+buckets, retreat,
type/weakness/resistance one-hots from `data/EN_Card_Data.csv`) behind a
**zero-init, bias-free projection** added to the entity+option card embeddings,
gated by `ModelConfig.use_card_meta` (mirrors `use_option_rank`; docs §0/§5
amended). Two load-bearing design choices: the projection is created *last* (a
meta-ON net consumes the identical RNG stream) and zero-init, so grafting the
pathway onto a meta-OFF parent via `--init-ckpt --use-card-meta`
(strict=False; missing keys = exactly the two meta tensors) is
**behavior-identical at iter 0** — metadata influence is *learned* during the
fine-tune, never injected as init noise. The table is a persistent buffer, so
checkpoints carry it (Kaggle bundle needs no CSV).

**Setup** (`scripts/run_coevo_meta_kill.sh`; same recipe as the judge_swap kill
run): deck = `agent/decks/archaludon_genesect_swap.csv` — parent Archaludon deck
with 1x Relicanth → **1x Genesect ex (547)**, chosen because it is (a) cold:
absent from ALL six training decks (the parent's whole training exposure is just
68 distinct cards), (b) coherent: Metallic Signal searches Evolution cards and
the deck runs 4x Duraludon → 4x Archaludon ex, (c) a *Pokémon*, so the metadata
actually describes most of what matters (220 HP Basic {M} ex, weak {R}) — for a
novel Trainer, metadata only says "Item/Supporter" (effect text is not encoded),
a known limit of the feature. Parent = `probe_archaludon_medium_long/best.pt`
(meta-OFF). **Parent-arm decision:** warm-ON grafts a fresh zero-init projection
onto the meta-OFF parent (local CPU); we did *not* first re-train a meta-ON
parent (needs Colab/L4) — see caveats. Both arms: 30 iters, 48 g/iter, LR
3e-5→1e-5, mixed_pool league, pool gate (both promoted, final gate 68.4 vs
68.5 — the low-n gate can't separate them; the high-n eval below can).

**Results** (160 games/opp side-swapped, Wilson 95% CI; ref = parent on
original deck from `outputs/coevo_kill/eval_ref.csv`; new CSVs in
`outputs/coevo_meta/`; logs `rl_research/coevo_meta_*.log`):

| Opponent | ref (orig) | zero-shot (novel) | warm-OFF | warm-ON | ON−OFF |
|---|---|---|---|---|---|
| kaggle:archaludon | 67.5% | 55.0% | 61.9% | 64.4% | +2.5 |
| kaggle:starmie | 91.9% | 87.5% | 83.8% | **92.5%** | **+8.7** |
| kaggle:dragapult | 73.8% | 57.5% | 66.2% | 66.9% | +0.7 |
| kaggle:alakazam | 54.4% | 51.9% | 52.5% | **65.6%** | **+13.1** |
| kaggle:romanrozen | 88.8% | 71.2% | 80.6% | 84.4% | +3.8 |
| heuristic | 86.2% | 80.6% | 79.4% | 84.4% | +5.0 |
| random | 100% | 99.4% | 99.4% | 100.0% | +0.6 |
| **mean** | **80.4%** | **71.9%** | **74.8%** | **79.7%** | **+4.9** |

**Reading the 2×2 — every cell behaved as predicted:**
1. **Zero-shot craters on a novel card** (−8.5pp vs ref; the judge swap cost
   only −2.4pp). Confirms the cold-embedding failure mode is real, and bounds
   the zero-shot pre-filter: it *cannot* rate novel-card mutations.
2. **Warm-OFF only half-recovers** (74.8%, still −5.6pp below ref after the
   same 30-iter budget that fully recovered the judge swap). Warm-start alone
   does not scale to novel cards at this budget.
3. **Warm-ON recovers to parity with ref** (79.7 vs 80.4) — the metadata
   feature rescues the case. ON−OFF = +4.9pp on 1120 games/arm (z≈2.8,
   p≈0.005); concentrated again in Alakazam (+13.1pp, z≈2.4) with warm-ON
   *exceeding* ref there (65.6 vs 54.4) — same latent-value signature as the
   judge run: fine-tuning finds matchup gains zero-shot can't see, and
   metadata is what lets it find them for a card it never trained on.

**VERDICT: PASS.** The coevolution search is no longer bounded to
near-neighbor swaps of already-seen cards — with `use_card_meta` the
warm-start fitness signal survives novel-card mutations. Adopted: **search
children train with `--use-card-meta`** grafted at warm-start.

**Caveats (honesty budget):** one seed per arm; the graft arm conflates "novel
card gets metadata" with "all cards get metadata during fine-tune" (starmie
+8.7 suggests some of the gain is general, not Genesect-specific); the
definitive setup is a **meta-ON parent** (metadata pathway already trained →
novel card slots into a mature representation) — worth a Colab re-train of the
Archaludon parent before scaling the search; and metadata stays blind to
Trainer effect text, so novel-Trainer mutations remain the weak spot (the
two-stage protocol should route those through warm-start, never zero-shot).
Per [[unproven-work-stays-out-of-writeup]] this stays a dated log entry until
the full search produces a deck that wins on the ladder.

### 2026-07-01 — PFSP hardening run validated: it400 dominates the submitted checkpoint

The `archaludon_alakazam_hardened` run (warm-start from the submitted
`probe_archaludon_medium_long/best.pt` it850, PFSP[var] + `model:alakazam_rl`
@2.0 in the league, 400/500 iters before we reclaimed the L4) directly targeted
the two ladder losses from the first RL sub (Alakazam 0-3, Starmie 0-2):

- **Alakazam matchup transformed**: eval vs `kaggle:alakazam` 60.8% → ~80%
  (86.7% at it375, 78.3% at it400, n=120 each); gate vs scripted alakazam
  52% → 85-87%. Vs the *trained* `alakazam_rl` exploiter: 24% → ~40%
  (plateaued it150+ despite PFSP feeding it the most games — a real strength
  ceiling vs that policy, not under-sampling).
- **Starmie held at 88-94% all run** — the tempo weakness is gone.
- **Head-to-head vs the submitted ckpt (local, 600g side-swapped, greedy):
  98.3% [97.0, 99.1], zero seat asymmetry.** Sanity-checked: the parent file
  is byte-identical to Drive's and plays competently vs the local heuristic
  (56% vs child's 61%, overlapping CIs) — so the blowout is real exploitation
  of the frozen parent it trained against, not a broken opponent.

**Honest read for the writeup**: the 98.3% is greedy-vs-greedy exploitation of
a deterministic sibling, *not* +98% ladder strength — on the neutral local
heuristic yardstick the child is ≈parity with the parent. The ladder case
rests on the matchup table: it patches exactly the "over-reliance on specific
matchups" risk the rubric penalizes. Also logged: the adaptive entropy coef
sat pinned at 0.0400 for all 400 iters (controller saturated — inspect before
the next long run); pool gate 60.1% → 69.4% with the final it400 gate the
best of the run.

**Decision: submit it400 as the new Archaludon agent** (replaces the 1055.9
sub as one of the two counting slots).

### 2026-07-02 — Ladder scout: hardening worked; new weak spots are field-Archaludon and Snorlax stall

Scouted both live subs (35 recent replays each, one time window — small n,
per-matchup cells smaller still):

- **Archaludon hardened it400 (sub 54253730, LB 963.8): 22-13 (63%).** The
  hardening run's targets are confirmed patched on the real ladder: Starmie
  3-0 (was 0-2) and Alakazam 2-2 (was 0-3). New top weakness is
  **non-mirror Archaludon pilots: 1-4 (20%, n=5)** — other people's
  Archaludon lists beat ours even though true mirrors go 2-2. Loss shape is
  ugly: 12 of 13 losses are blowouts (≥3 prizes still needed; 4 losses took
  *zero* prizes, two of them 39-86 steps) → setup fragility / it folds
  entirely when the opening whiffs, rather than losing close endgames.
- **Alakazam (sub 54235407, LB 966.4): 23-12 (66%).** Farming the
  Archaludon-heavy field: 12-4 (75%) vs Archaludon, which is 46% of its
  games — great meta positioning, consistent with the local 17-22 edge.
  New hard counter: **Hop's Snorlax 0-3**, all long grinds (126-188 steps);
  4 of its 12 losses are deckouts. That's a stall/mill archetype the agent
  has never trained against — it wins races but can't win attrition.
- **Cross-agent shape**: our Alakazam beats field Archaludon; field Alakazam
  is now only even vs our Archaludon (was a loss). The pair hedges well.
  Losses avg 148 steps for Alakazam (late-game problem) vs 120 for
  Archaludon (early-game problem) — the two agents fail in opposite phases.

**Follow-up (same day)**: diffed the non-mirror Archaludon lists in the
cached replays — they are our list ± 1-2 cards (Judge over Boss's Orders /
Pokégear; one runs 2x Xerosic's Machinations). So "different lists" is NOT
the story: combined pseudo-mirror record is 3-6 (33%, n=9), and the losses
are pilot-skill/variance in an effectively identical-deck matchup, not deck
coverage. Two of the opposing lists swap in **Judge** (hand disruption),
which composes badly with our 0-prize setup-collapse losses.

**Fix directions**: (1) add a Hop's Snorlax scripted/exploiter opponent to
the Alakazam pool (stall exposure + deckout awareness); (2) for Archaludon,
the mirror gap is play-strength not list coverage — investigate the 0-prize
collapses (replays under `outputs/replays/scout/54253730/`), consider
hand-disruption (Judge) exposure in the pool, and possibly test the Judge
swap in our own list.

**Implemented (same day) — ladder swaps wired into the meta-ON run.** The
three observed variant lists were extracted verbatim from the replays into
`agent/decks/archaludon_ladder_{judge,judge_metal,xerosic}.csv` (asserted
exact match to the observed multisets; engine-validated via
`validate_deck.py`). The meta-ON lineage-root cell in
`colab_selfplay_archaludon.ipynb` now adds two extra `--league-checkpoint`
opponents — the hardened-it400 root itself piloting the judge list @1.5 and
the xerosic list @0.75 (fixed ROOT, not the resume PARENT, so sparring stays
constant across crash-resumes) — giving the learner its first exposure to
hand disruption from a strong near-mirror pilot. Watch the `arch_judge` /
`arch_xerosic` buckets in the gate breakdown. The same three CSVs double as
**generation-0 seeded candidates** for the coevo deck search; since all
three swap Trainers (metadata-blind), they route straight to warm-start,
never the zero-shot pre-filter (rule + table added to
COEVOLUTIONARY_DECK_SEARCH.md §3). Verified end-to-end with a 2-iter CPU
smoke run: card-meta graft loads (`missing=[card_meta_table,
card_meta_proj.weight]`), both new league entries register, and the mix log
shows the variant opponent being sampled.

<!-- Append new dated entries above this line as strategy evolves. -->
