# First RL agent on the ladder — replay analysis (submission 54211098)

Date: **2026-06-30**. First self-play RL checkpoint fielded on Kaggle: **Archaludon, medium
(15.8M), it~300/1200** of the scaled long-run (see
[`PROBE_ARCHALUDON_2026-06-30.md`](./PROBE_ARCHALUDON_2026-06-30.md)). Packaged with
`scripts/build_rl_submission.py` (Kaggle-faithful post-build gate; the first attempt errored
on a module-level `__file__` under the runtime's `exec()` — now guarded). Analysis over **32 of
62 public episodes** (win/loss from `rewards[our_seat]`, opponent archetype from
`steps[1][opp].action`, prize race from per-step `observation.current.players[].prize`).

## Result: 18W–14L (56%), rating **987.4** (vs our v1 heuristic 754.4)

Above the heuristic, and climbed from the 600 baseline — but with two exploitable failure modes.

### Matchup table (32 games, always piloting Archaludon)
| opponent archetype | record | note |
|---|---|---|
| Archaludon ex (mirror) | **9-3** | strongest — the self-play matchup |
| Mega Lucario ex | **5-2** | strong — heavily in the pool (romanrozen) |
| Cynthia's Garchomp ex | 2-1 | fine |
| **Alakazam** (mislabeled "Fezandipiti ex") | **0-3** | **hard counter, NOT in training pool** |
| **Mega Starmie ex** | **0-2** | lose the tempo race (Starmie *is* in the pool) |
| Dragapult ex | 0-1 | long grind loss (bench spread) |
| Cornerstone Ogerpon ex | 0-1 | long grind loss (wall) |
| Great Tusk | 0-1 | fast aggro |
| Hop's Phantump / Mega Kangaskhan | 1-0 each | — |

**Label correction:** the "Fezandipiti ex" row is actually **Alakazam** (Abra/Kadabra/Alakazam +
Dunsparce/Dudunsparce; Fezandipiti ex is a 1-of tech the classifier latched onto). Alakazam is
the real weakness.

## How it loses — prize-race evidence (prizes *remaining*; first to 0 wins)
- **Alakazam blowout** (ep 82914177, 79 steps): opp 6→5→4 while **we never took a single prize
  (stuck at 6)**. Alakazam's hand-size-scaling OHKOs ran us over before setup.
- **Alakazam grind** (ep 82909681, 151 steps): closer, lost the race 2→1.
- **Starmie tempo** (ep 82916812): opp 6→**4** while we're still at **6**; claw to 3, lose.
  (ep 82921777): opp 6→**3** while we're at **6**; lose 2→1. Starmie (Cinderace accel + cheap
  Jetting Blow) attacks faster than our 3-energy Metal Defender plan — we spot 2–3 early prizes.
- **Mirror win vs loss:** the WIN (ep 82929175, 171 steps) was a patient grind to 2→1; the LOSS
  (ep 82926165) is where we fell behind early (opp at 3 while we're at 6 by turn 7).

## Two failure modes
1. **Outraced by aggro** — Alakazam OHKO, Starmie tempo, Great Tusk. We set up slowly and spot
   early prizes we can't recover.
2. **Out-attritioned by spread/wall** — Dragapult Phantom-Dive, Ogerpon Cornerstone (long grinds
   we lose).

Median win length (138 steps) ≈ loss length (132) — losses aren't "too slow" games, they're
games where we **fell behind early and couldn't catch up**.

## Root cause — the deck-resilience gap, confirmed live
Strong vs trained/slow decks (mirror, Lucario), exploited by fast aggro it never trained against.
Pure self-play + a narrow pool overfit to the mirror and the few pool archetypes. This is exactly
what [[orbit-wars-selfplay-lessons]] and [[bc-pool-plan]] predicted; the ladder just proved it
with money. Note **Starmie was in the pool at weight 1.5 yet we still went 0-2** — the pool's
Starmie (our from-scratch rule agent) is weaker/slower than the ladder's Starmie pilots, so it
didn't teach the fast matchup. Pool *quality/diversity*, not just presence, matters.

## Next actions (priority order)
1. **Add Alakazam to the training pool** — biggest, cheapest gap-closer (0-3 blind spot;
   `ryotasueyoshi` Alakazam kernel already in `outputs/kernels/`). One vendored agent + manifest
   line, same as the Starmie add ([[asymmetric-selfplay-infra]]).
2. **Fix the Starmie tempo gap** — up-weight it, add a faster Starmie pilot, and/or investigate
   whether the policy under-values early pressure (reward/curriculum).
3. **PFSP + exploiters + BC-clones of ladder aggro** — BC the exact replay opponents
   (Alakazam/Starmie/Great Tusk) into the pool ([[bc-pool-plan]]); add an exploiter that hunts
   this outraced-by-aggro weakness.
4. **The long-run final checkpoint** will be stronger but carry the *same* blind spots unless the
   pool is widened first — widen, then submit the final.

Reusable mechanics now in place: `scripts/build_rl_submission.py` (+ gate), `agent/rl_main.py`,
and the replay-analysis flow (`kaggle_tool.py episodes/replay` → prize-race + archetype parse).
