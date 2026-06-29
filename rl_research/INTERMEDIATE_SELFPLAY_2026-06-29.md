# Intermediate self-play — Agents A & B on the asymmetric, deck-agnostic league

Date: **2026-06-29**. Companion to [`DECK_SELECTION_STRATEGY.md`](./DECK_SELECTION_STRATEGY.md)
(the *which-deck* decision + flexibility mandate) and [`SELFPLAY_RL_PLAN.md`](./SELFPLAY_RL_PLAN.md).
This is the **dated experiment record** for the first runs on the pivot-ready
infrastructure: cheap, intermediate, *not* the final big-budget runs (those lock in
early August). Goal = (1) prove the deck-agnostic asymmetric pipeline learns, and
(2) start **banking checkpoints** as opponent-pool assets that survive any deck pivot.

## What shipped (the pivot-ready foundation)

Everything below takes the deck(s) + opponent set as **data/params** — swapping a
deck or a Kaggle opponent is a config change, never a code change (the flexibility
mandate). Nothing hardcodes A=Archaludon / B=Dragapult.

- **Asymmetric engine seating.** `dist_worker` now calls
  `battle_start(deck_self, opp_deck)` (was `battle_start(deck, deck)`): the trainee
  deck always seats the model; the opponent's own deck arrives per game in the PLAY
  token. `self`/`model:` opponents pass `opp_deck=None` → mirror the trainee deck;
  fixed/Kaggle opponents pilot **their own** deck. Seat side-swap (model_seat = g%2)
  still cancels the ~51.5% first-player edge (the deck can't react to play order, so
  the *policy* must handle both seats).
- **Deck-carrying league.** `dist_collector.League` gained a `decks: {spec → deck}`
  map; `build_league`/`league_fixed_specs` take `extra` opponents + `opp_decks`. The
  central GPU loop still trains **only** the `cur` policy's decisions — a `model:`/
  Kaggle opponent contributes pressure but no gradient (preserved by construction).
- **Opponent manifest = league as DATA.** `src/ptcg_battle/opponents.py` +
  `agent/opponents/mixed_pool.json`. Each opponent is `{agent, weight, deck?}`;
  `kaggle:<name>` auto-resolves its own deck from `agent/kaggle_agents/<name>_deck.csv`.
  Add an archetype by dropping a vendored module + sibling deck and adding one JSON
  line.
- **Parameterized train/eval/build.** `train_selfplay.py --deck <trainee.csv>
  --opp-manifest <pool.json>`; `eval.py --deck <champion.csv>` + `kaggle:<name>`
  opponents (each pilots its own deck); `build_submission.py --deck` already took a
  path. `quick_eval`/`play_match`/`load_deck` already deck-parameterized.
- **Borrowed opponents carry their own decks.** The bug that bit us before: each
  Kaggle kernel reads `deck.csv` from cwd at import, and workers `chdir(agent/)`, so
  two borrowed agents would both load the *trainee* deck and collide. Fixed: every
  vendored agent now reads a **sibling** `<name>_deck.csv` relative to its own file.

### Opponent pool (`agent/opponents/mixed_pool.json`), validated

`scripts/validate_kaggle_agent.py <name>` plays each through a full self-game,
asserting a legal 60-card deck + in-bounds selections every decision (the submission
contract). All pass.

| spec | archetype | deck (own) | source kernel |
|------|-----------|-----------|---------------|
| `kaggle:archaludon` | Metal control + Cinderace accel | `archaludon_deck.csv` | masamikobayashi *a-sample-archaludon* |
| `kaggle:dragapult` | Phantom-Dive spread/combo | `dragapult_deck.csv` | kiyotah *sample rule-based (Dragapult ex)* |
| `kaggle:romanrozen_v10` | Fighting / Mega-Lucario | `romanrozen_v10_deck.csv` | romanrozen *STRONG START V10 (LB 950+)* |
| `heuristic` | Lucario (our `main.py`) | `agent/deck.csv` | ours |
| `random` | deck-agnostic noise | mirror | — |

Plus `self` + up to 5 frozen **past checkpoints** (added by the trainer, weight
`w_past=2.0` split evenly). skarin's *Phantom Dive or Go Home* Dragapult kernel was
also pulled (`outputs/kernels/skarin__…`) for reference; kiyotah's cleaner sample is
the vendored Dragapult opponent.

## Runs (intermediate; CPU, 8-core box, no CUDA here)

Both: `--size small (5.7M) --league --collector dist --workers 6 --iters 20
--games-per-iter 32 --epochs 3 --minibatch 256 --lr 3e-4→1e-4 --target-entropy 0.15
--snapshot-every 4 --pool-size 5`, opponent pool = `mixed_pool.json`. The trainee
faces its own deck (self/past mirrors) **and** the cross-archetype Kaggle agents on
*their* decks — i.e. genuinely asymmetric games (Archaludon-trainee vs Dragapult-on-
Dragapult, etc.). Throughput here is CPU-bound (~1 game/s, ~0.5–3 min/iter as the
pool + game length grow); **the real runs belong on the L4/Colab** (Phase-0 finding —
GPU batched inference needs batch ≥48). These runs validate the pipeline + bank
assets, nothing more.

### Agent A — Archaludon (`agent/decks/archaludon.csv`)

In-loop eval vs `random` (side-swapped, n=40, greedy): **best 92.5% @ it5**, 90% @
it10. Banked: `outputs/checkpoints/intermediate_archaludon/{best,last}.pt`. The
late-iter vs-random number drifts down (72.5% @ it20) as the adaptive-entropy
controller (target 0.15) re-injects exploration on the small per-iter budget — a
known small-n artifact, not a regression of `best.pt` (which is frozen at it5). The
PPO health is clean throughout (KL under the 1.5 cut after it2, clipfrac falling,
value loss ~0.05).

### Agent B — Dragapult (`agent/decks/dragapult.csv`)

In-loop eval vs `random`: **best 77.5% @ it5**, 80% @ it10. Banked:
`outputs/checkpoints/intermediate_dragapult/{best,last}.pt`. Dragapult games run
~2× longer (≈3k decisions/iter vs ≈1.5k for Archaludon — more setup/evolve options),
so iters are slower; the lower vs-random ceiling at this budget is consistent with
the deck's higher decision complexity needing more samples, *not* a pipeline issue.

### Matchup eval vs the pool (`best.pt`, greedy, side-swapped, Wilson 95% CI)

`scripts/eval.py --champion model:…/best.pt --deck <trainee> --opponents
mirror,random,kaggle:<foil>,kaggle:romanrozen_v10` — each Kaggle opponent pilots its
own deck (true asymmetric matchup). `mirror` = identical copy of the champion (the
A/A null).

**Agent A — Archaludon** `best.pt` (n=80/opp, side-swapped):

| opponent | win rate | 95% CI |
|----------|---------:|--------|
| `mirror` (A/A null) | 46.2% | [35.7, 57.1] |
| `random` | 88.8% | [80.0, 94.0] |
| `kaggle:dragapult` | 7.5% | [3.5, 15.4] |
| `kaggle:romanrozen_v10` | 5.0% | [2.0, 12.2] |

**Agent B — Dragapult** `best.pt` (n=60/opp, side-swapped):

| opponent | win rate | 95% CI |
|----------|---------:|--------|
| `mirror` (A/A null) | 51.7% | [39.3, 63.8] |
| `random` | 75.0% | [62.8, 84.2] |
| `kaggle:archaludon` | 3.3% | [0.9, 11.4] |
| `kaggle:romanrozen_v10` | 5.0% | [1.7, 13.7] |

**Read:** both A/A nulls sit at ~50% (±10–12pp at this n) → the side-swapped eval is
unbiased; both policies **decisively beat `random`** (the "does it learn?" gate —
yes); both get **crushed by the hand-tuned rule-based agents** (~3–8%). That gap is
*expected and the headline finding*: 20 iters of a 5.7M model on **CPU** is nowhere
near the strong public heuristics — strength needs the L4 budget (more iters, larger
model, GPU-batched collection). The infrastructure is validated; competitiveness is
a compute problem, exactly as the Orbit Wars lesson predicted (bet on throughput +
scale, not a quick CPU run). These checkpoints are banked as **opponent-pool assets**
regardless — frozen, diverse pressure for the August big-budget runs, deck pivot or
not.

## Banked assets (keep — they survive a deck pivot)

- `outputs/checkpoints/intermediate_{archaludon,dragapult}/{best,last}.pt` (gitignored;
  push to Drive per the Colab workflow). Every checkpoint — even on a deck we later
  abandon — is free, diverse pressure in the opponent pool and a seed for fast
  per-deck fine-tunes after a pivot.
- Logs: `rl_research/logs/intermediate_{archaludon,dragapult}.log` (committed record).

## Takeaways / next

1. **Pipeline learns on an arbitrary trainee deck via the asymmetric league** — the
   deck-agnostic foundation works end to end (vendor → manifest → asymmetric collect
   → PPO → checkpoint → deck-parameterized eval).
2. These are **CPU validation runs**; promote the same commands to the L4/Colab
   (`--device cuda`, more workers, ≥100 iters, larger `--games-per-iter`) for real
   strength. Keep `--target-entropy` lower or fix `--ent-final` decay for the final
   runs so late iters don't re-explore away a good policy at small budgets.
3. **Breadth is one JSON line away.** Add Starmie / Alakazam / Crustle opponents by
   vendoring their kernels (`ryotasueyoshi` Alakazam, a Crustle wall kernel are
   already in `outputs/kernels/`) + sibling decks, then adding manifest entries — no
   code change. Do this before the August lock to widen the robustness pressure.
   **DONE for Starmie (2026-06-29):** see "Pool addition" below. Still TODO: Alakazam,
   Froslass, Crustle.
4. Re-pull the leaderboard weekly and update the pool weights toward the live
   top-band distribution (the deck-selection doc's recency-weighted objective).

## Pool addition — `kaggle:starmie` (Mega Starmie ex + Cinderace), 2026-06-29

Closed the biggest meta gap: **Mega Starmie ex is the #2 top-band deck** (6/34 seats,
[[meta-snapshot-2026-06-29]]) yet was absent from the training pool, so A (Archaludon)
was scaling without ever facing the second-most-common opponent.

**Source sieve.** The two public Kaggle Starmie kernels publish **no full agent**:
masamikobayashi *"Prize Card Tracking: 1300+ Starmie"* (gold-medal write-up; shares only
a `PrizeTracker` helper) and map1e114514 *"Starmie Cinderace Budew Skill Agent"* (a
`skills/` architecture write-up, no code). So `agent/kaggle_agents/starmie.py` is a
from-scratch rule engine on the repo's own clean scoring scaffold (mirrors `archaludon.py`'s
generic board helpers + score dispatcher), tuned to the engine's **authoritative**
card/attack data, with the game plan informed by both write-ups. Pilots our
replay-extracted, engine-validated top-ladder list (`agent/decks/mega_starmie.csv` →
sibling `starmie_deck.csv`).

**Game plan encoded:** Cinderace Explosiveness (face-down Active in setup) → Turbo Flare
([C]=50, accelerate 3 Basic Energy from deck to Bench) → evolve Staryu into Mega Starmie ex
via Mega Signal / Salvatore / Wally → attack with Jetting Blow ([W]=120 +50 bench snipe) or
Nebula Beam ([C][C][C]=210, ignores Weakness/effects), with Crushing Hammer + Boss
disruption. No Budew in this list (the public lists run it; ours doesn't), so no item-lock
sub-plan. Crash-safe wrapper + `validate_kaggle_agent.py starmie` pass (legal deck + legal
selections every decision).

**Sanity eval** (`kaggle:starmie` champion, side-swapped, n=60/opp, CPU, greedy):

| opponent | Starmie win rate | 95% CI | read |
|----------|-----------------:|--------|------|
| `random` | 93.3% | [84.1, 97.4] | plays the deck correctly, not just legally |
| `kaggle:dragapult` | 53.3% | [40.9, 65.4] | near-even / competitive |
| `kaggle:archaludon` | 18.3% | [10.6, 29.9] | loses to its **designed counter** (that kernel = "75% WR vs my 1300+ Starmie") — the real-world matchup, not a bug |

This is a genuinely competent, archetype-accurate Starmie — far above the 5.7M CPU
checkpoints (3–8% vs every rule agent). We deliberately did **not** tune Starmie to beat
its hard counter; the archetype + validated deck are the pressure. Wired into
`agent/opponents/mixed_pool.json` at **weight 1.5** (equal to the other strong archetypes,
reflecting its #2 meta standing). Now there is enough deck diversity in the pool — Metal
control, Water tempo, Dragon spread, Fighting/Lucario — to justify scaling A/B on the L4.
