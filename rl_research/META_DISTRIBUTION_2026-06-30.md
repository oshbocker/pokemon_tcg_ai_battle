# Ladder meta distribution — deck & strategy snapshot (2026-06-30)

Sample: **226 replays** (452 seats, 71 distinct decklists) = a fresh pull of the top-20 teams'
top-submission episodes (`scripts/pull_top_replays.py 20 6`) + the prior `outputs/replays/` cache.
Reproduce with **`uv run python scripts/analyze_meta_replays.py`** (now reports seat% AND
distinct-pilot%, merges the Alakazam-combo labels, and skips errored games in the WR denominator).

**Read the `pilot%` column, not `seat%`.** `seats` is raw appearances — inflated by repeated games,
notably our own ~30 cached RL-agent (Archaludon) episodes. `pilots` = distinct 60-card lists, the
truer "how many teams play this" signal.

## Deck distribution

| Archetype | seat% | **pilot%** | WR% (decisive) |
|---|---|---|---|
| **Alakazam combo** (hand-scaling OHKO) | 28 | **32** | 51 |
| **Archaludon ex** (Metal control/grind) | 32\* | **13** | 49 |
| **Mega Starmie ex** (Water tempo) | 9 | **10** | 43 |
| **Mega Lucario ex** (Fighting aggro) | 4 | **10** | 47 |
| **Marnie's Grimmsnarl ex** (Dark aggro) | 8 | **7** | 57 |
| Hop's Snorlax (single-prize aggro) | 2 | 4 | 33 |
| Cynthia's Garchomp ex (Dragon aggro) | 2 | 4 | 57 |
| Dragapult ex (spread) | 4 | 4 | 47 |
| Chandelure (spread/burn) | 3 | 3 | 60 |
| Cornerstone Ogerpon ex (wall/stall) | 2 | 3 | 62 |
| Mega Kangaskhan ex (Colorless aggro) | 2 | 3 | 45 |
| TR Mewtwo ex / Crustle / Froslass / Clefairy / Heracross | tail | 1 ea | — |

\* Archaludon's seat% is inflated by our own cached episodes; its real footprint is the 13% pilot figure.

## Strategy axis (by distinct pilots)

- **Combo aggro (Alakazam) ~32%** — the single dominant deck.
- **Assorted aggro ~28%** — Lucario (10) + Grimmsnarl (7) + Garchomp (4) + Snorlax (4) + Kangaskhan (3).
- **Control/grind ~13%** — Archaludon.
- **Fast tempo ~10%** — Starmie.
- **Spread ~7%** — Dragapult + Chandelure.
- **Wall/disruption ~5%** — Ogerpon + Crustle + Froslass.

Aggro of some flavor is the dominant strategy class; the field is ~1/3 Alakazam combo + a broad aggro long tail.

## The headline: Alakazam combo is #1, and it beats us

The HP-weighted namer split one physical deck into "Fezandipiti ex" / "Alakazam" / "Dudunsparce"
by whichever 1-of ex it saw — but **100% of all three carry the Abra/Kadabra/Alakazam line
(741/742/743)**, so they're one deck: the hand-scaling OHKO combo (Powerful Hand = 20 dmg × cards
in hand). Merged, it's **the most-played archetype on the ladder (~32% of pilots)**.

Head-to-head, **Archaludon (our trainee deck) goes 17-22 (44%) vs Alakazam combo** over n=39 — the
biggest matchup in the sample and a losing one. This corroborates the first RL submission's 0-3 vs
Alakazam ([[rl-agent-ladder-weaknesses]], `RL_LADDER_ANALYSIS_2026-06-30.md`) on a much larger n:
it's a real, structural bad matchup, not variance. Even the hand-coded Archaludon rule agent loses it.

## WRs cluster near 50% → prevalence, not win rate, defines the meta

Skill-rated ladders self-balance, so no deck runs away on WR. The few decks sitting >50% are the
ones **not in our training pool**: Ogerpon wall (62), Chandelure (60), Grimmsnarl (57), Garchomp (57).

## Implications for the training pool

Current pool (`agent/opponents/mixed_pool.json`): archaludon, starmie, dragapult, **alakazam (added
today)**, romanrozen/Lucario, heuristic, random — now covers the **top 4–5 archetypes by pilot share**.
Adding Alakazam is validated as the #1 gap-closer by this larger sample. Next candidates, in order,
are the >50%-WR decks we still don't train against:

1. **Marnie's Grimmsnarl ex** — biggest missing (7% of pilots, 57% WR, Dark aggro).
2. **Cynthia's Garchomp ex** (4%, 57%) and **Chandelure** spread/burn (3%, 60%).
3. **Cornerstone Ogerpon ex** wall (3%, 62%) — a different axis (stall) our pool lacks entirely.

## Caveats

Top-20-team-weighted sample (opponents = whoever the ladder matched them against — representative but
tilted toward what strong teams face). Strategy-axis labels are hand-mapped. Small-n on the tail
matchups. Archetype namer keys on highest-HP ex + the two signature overrides (Crustle line, Alakazam line).
