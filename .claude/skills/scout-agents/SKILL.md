---
name: scout-agents
description: Pull recent Kaggle ladder replays for our own submitted agents and report per-agent strengths/weaknesses (win-rate, per-opponent matchup table, loss shape). Use when asked to scout/analyze our live agents, audit how our submissions are doing on the ladder, or find matchup weaknesses to fix.
---

# Scout our live agents

Downloads recent ranked replays for our Kaggle submissions and reports, per
agent, how it's performing on the live ladder: overall win-rate, a per-opponent
matchup table, and the "shape" of its losses (game length + prize margin +
deckouts) so you can tell blowouts from close games and find the matchups worth
fixing.

## How our seat is identified

Every replay records both players' full 60-card decks at `steps[1][seat].action`.
Our deck is the same in every one of a submission's games, so **our seat is the
60-card fingerprint that is constant across all of that submission's replays** —
no guessing, and it's robust even when the opponent plays the same archetype
(those show up as `(MIRROR)` rows). Opponent archetypes are named with the same
main-attacker logic as `scripts/analyze_meta_replays.py`.

## Usage

Everything is in `scripts/scout_agents.py` (fetch + analyze in one shot; replays
cache under `outputs/replays/scout/<submission_id>/` and are not re-downloaded).

```bash
# Default: latest 2 COMPLETE submissions, 35 recent replays each
uv run python scripts/scout_agents.py

# More depth
uv run python scripts/scout_agents.py --n 3 --replays 50

# Specific submissions (find ids: uv run python scripts/kaggle_tool.py submissions,
# or the printed `ref` — see scripts/kaggle_tool.py)
uv run python scripts/scout_agents.py --sub 54219624 --sub 54235407

# Re-analyze the cache without hitting Kaggle
uv run python scripts/scout_agents.py --no-fetch --sub 54219624 --sub 54235407
```

Requires `~/.kaggle/kaggle.json` and `data/EN_Card_Data.csv` (card metadata:
`uv run python scripts/kaggle_tool.py download-data`).

## Reading the output & writing up

For each submission the script prints: record + decisive WR, avg game length in
wins vs losses, a loss-margin histogram (prizes we still needed — `>=3` = blowout,
`1` = one prize away), the per-opponent W-L-D/WR table, and the full loss list.

When the user asks for analysis, don't just paste the numbers — synthesize:
- **Strengths** = high-WR / high-n opponent buckets and whether wins are fast
  (low step count).
- **Weaknesses** = low-WR buckets, weighted by how often that opponent appears
  (a 15% matchup that is 1/3 of your games is the top priority).
- **Loss shape** = blowouts (took 0-1 prizes) point at *setup fragility /
  training-pool coverage gaps*; close losses point at *late-game tactics*;
  deckouts/very-long games point at *stall/mill* matchups.
- **Cross-agent** = compare our two agents against each other and note any
  rock-paper-scissors, and whether each sits above/below the field (~50% WR).
- Tie weaknesses back to concrete fixes: add the problem archetype as an
  **exploiter / to the self-play opponent pool**, or reconsider the deck.

If the finding is strategically meaningful, append a dated entry to
`rl_research/STRATEGY_WRITEUP_LOG.md` (the living writeup log). Caveat the sample
size (n per agent, one time window) and note that `(MIRROR)` rows are
self-play/identical-deck opponents and low-signal.
