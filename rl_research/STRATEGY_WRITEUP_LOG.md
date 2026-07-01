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

<!-- Append new dated entries above this line as strategy evolves. -->
