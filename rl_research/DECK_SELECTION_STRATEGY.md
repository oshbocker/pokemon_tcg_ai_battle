# Deck-Selection Strategy — the meta-game above the RL

Status: **decided framing** (2026-06-29). Companion to
[`SELFPLAY_RL_PLAN.md`](./SELFPLAY_RL_PLAN.md) (the *how-to-train* doc) and
[`LESSONS_FROM_ORBIT_WARS.md`](./LESSONS_FROM_ORBIT_WARS.md). This doc is the
*which-deck-and-why* decision that sits **above** the RL training problem and
largely determines whether the compute we spend on training pays off.

We split the problem into two optimizations:
1. **Deck selection** (this doc) — pick the deck(s) that maximize expected
   leaderboard rank against the field we will face.
2. **Agent training** (SELFPLAY_RL_PLAN) — given a deck, train the best pilot.

**Deck selection dominates.** A perfectly-trained agent on a countered deck loses
to a moderately-trained agent on the right deck. Lucario was Tier-0 on 2026-06-20
and is ~1/34 at the top by 2026-06-29 (see [meta snapshot] / replay extraction).
So this decision is first-order; compute allocation is second-order.

## Hard competition mechanics that drive everything

(Verified from `README.md` "Evaluation & ranking" + the engine, 2026-06-29.)

- **Only the latest 2 submissions count for final evaluation.** We effectively
  *field 2 agents*. (5 submissions/day allowed; LB shows only our best one.)
- **Matchmaking is rating-banded** — *N(μ, σ²)*, paired with *similar* ratings.
  We do **not** face the global field uniformly, nor the top, until we *are* near
  the top. The field we're scored against = the **top band**, *conditional on our
  agent being good enough to climb there*.
- **Win/loss only** — margin of victory is irrelevant. Optimize win *probability*.
- **σ shrinks with games; ladder keeps running Aug 17–31** until convergence.
  Submission **timing is a resource** (don't thrash σ near the deadline).
- **All replays are public** → our exact deck and lines are scoutable. We are
  transparent. (We exploit this too: `scripts/extract_decks_from_replays.py`
  reads both decks from `steps[1][seat]["action"]`.)
- **Deck choice is blind and unconditional.** At deck-selection (`obs.select is
  None`) the observation is *empty* (`current=None, select=None, logs=[]`) and
  identical for both seats. We do **not** know our opponent, our rating gap, or
  even whether we play first/second (`firstPlayer` is still `-1` at turn 0,
  decided in setup *after* decks lock). No scout-and-counter, no play/draw hedge.

## Bottom-line findings

1. **The "multi-deck vs single-deck" dichotomy is a false frame.** Decompose:
   - *Opponent-deck diversity* in training → always worth it, **cheap**, strictly
     improves robustness. (This is the league/asymmetric-`battle_start` work.)
   - *Pilot-deck count* → should equal the number of agents we field (**~2**),
     each a **specialist**, not one omni-deck generalist. Even at infinite compute
     the optimum is a *population of specialists* (Nash over (deck,policy)), not one
     generalist — multi-task RL shows negative transfer. So fielding ~1 deck per
     agent is correct, and the leverage is in *which* deck.
2. **Objective:** maximize **E[win prob] against the top-band, recency-weighted
   opponent distribution**, since that's where final rank is decided. Keep
   re-pulling the leaderboard and updating priors (weekly).
3. **Prefer low-exploitability (maximin / flat matchup spread / execution-based)
   decks** over high-ceiling decks with a hard counter, and over surprise/
   contrarian decks. Rationale: the meta churns *and* replays make us transparent,
   so the durable edge is "we pilot it better when the opponent knows our deck,"
   not "they haven't seen it." No strictly dominant deck exists (RPS structure).
4. **Use the 2-agent slot as a hedge, not two copies of one bet:**
   - **Agent A (anchor):** best vs the current top band (max-expected or maximin).
   - **Agent B (cover):** counters A's *worst* matchup / the reigning Tier-0.
   - Value = variance reduction; LB rewards our best of the two.
   - **"Latest 2" forbids accumulating a portfolio** — experiment freely through
     July, then **lock the final 2 in early August** (late enough for end-meta,
     early enough for σ to converge).
5. **Contrarian / under-represented pick = tie-breaker only.** Weak here because
   bots don't sideboard and matchmaking is by rating (can't dodge counters). Real
   only where heuristics carry hard-coded tech keyed to popular decks (e.g. our
   own Crustle-aware routing) — prefer the candidate whose *counters* are rarest.
6. **Handle the play/draw asymmetry in the policy, not the deck** (deck can't react
   to it): sample both seats evenly in self-play; keep side-swapped eval.
7. **Shared-trunk + per-deck fine-tune: right idea, defer the commit.** Build the
   one-deck self-play agent first (Phase 0 still gates everything). Its trunk
   becomes the *optional* seed for fast per-deck fine-tunes **if/when the meta
   forces a pivot** — that agility (respawn a specialist quickly) is the real win,
   not multi-deck generality. Don't architect for n decks before n=1 works.

## Implication for *us, now*

Our existing asset is a **Lucario specialist on a fading deck.** The framework
says **re-pick before pouring compute into training** — most likely toward the
current low-exploitability top-band decks. Candidates surfaced by replay
extraction (2026-06-29 top ladder): **Archaludon ex (13/34), Mega Starmie ex
(6/34), Mega Froslass ex (3/34).** All recovered + engine-validated under
`agent/decks/*.csv`, alongside iono/crustle/cynthia_garchomp/marnie_grimmsnarl
and the tuned Lucario.

## Selected agents — working set (2026-06-29)

Picked from leaderboard **replay analysis** (`scripts/analyze_meta_replays.py` over
74 top-ladder replays), not a self-run round-robin. These are a **working set, not
a final lock** (see Flexibility below).

- **Agent A = Archaludon ex** (anchor). Current #1 (47/144 seats). Bulky single-ex
  Metal control (300 HP, weak Fire) behind heavy draw/disruption (Judge, Carmine,
  Lillie, Boss). Beats Starmie 6-3; even vs Alakazam. Edge is execution/consistency
  → durable under public-replay transparency. Deck: `agent/decks/archaludon.csv`.
- **Agent B = Dragapult ex** (complementary foil). Phantom Dive = 200 to active +
  60 spread on the bench — the hardest axis for a one-tank metal deck (snipes the
  bench where Archaludon sets up its next attacker), so it's the best *sparring*
  partner to robustify A. 320 HP, no weakness → long informative games. Maximal
  style contrast (combo/spread vs control); no shared failure mode with A (unlike
  Starmie, which A dominates and which shares A's "single ex → folds to Crustle"
  flaw). Richest public-agent support (kiyotah sample + skarin). Deck:
  `agent/decks/dragapult.csv`.
- **Agent C (candidate) = Alakazam** (non-ex). Best raw current WR in-sample (67%,
  even vs Archaludon, beats Starmie 3-0), non-ex resilience (dodges Crustle, cheaper
  prize trades) but fragile (140 HP, weak Dark), thin pool support, small sample.
  Hold as the next deck to spin up.

Matchup signal: Archaludon **loses** to the non-ex toolbox decks, Marnie's Grimmsnarl
ex (2-1), and Fire (Chandelure) — candidate covers if we need to hedge A's weak side.

## Flexibility & deck-pivot principle (explicit design constraint)

The meta churns (Lucario→Archaludon in 9 days); today's picks **will likely change.**
So **infrastructure and code must be deck-agnostic and pivot-ready** — never hard-wire
A=Archaludon / B=Dragapult. Concretely:

- Decks are data (`agent/decks/*.csv`), selected by name/flag — never hardcoded.
- The training collector, league, eval harness, and submission build all take the
  deck(s) + opponent-agent set as parameters (asymmetric `battle_start(deckA, deckB)`).
- Adding/swapping a deck or a Kaggle opponent agent is a config change, not a code
  change.
- **Two-phase plan:** *now → ~late July* run **cheaper intermediate training runs**
  on the working set, staying flexible; *early August* **lock the final ~2 agents**
  for the **big-budget runs** (σ-convergence vs end-meta tradeoff per the mechanics
  above).
- **Every intermediate agent is an asset, even if we pivot away from its deck:**
  - **Keep its trained weights** — past checkpoints (any deck) feed the opponent
    pool and robustify *all* future training (a frozen old self/other-deck agent is
    free, diverse pressure).
  - **Record findings** per agent (deck, training config, eval/matchup results,
    leaderboard outcome) in `rl_research/` so deck-pivot decisions are data-driven.
  - A trained general trunk is the seed for **fast per-deck fine-tunes** when we
    pivot — agility is the payoff of keeping everything.

## Next steps

1. Pull the skarin Dragapult agent into `agent/kaggle_agents/` for the pool.
2. Build the **deck-agnostic, asymmetric** training collector + league (params:
   our deck, opponent deck(s), opponent-agent set) — the pivot-ready foundation.
3. Run intermediate self-play on A and B against the mixed pool (self + past
   checkpoints + A↔B + Kaggle agents across archetypes); record findings; bank
   all checkpoints into the pool.
