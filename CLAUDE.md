# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

Kaggle **Pokémon TCG AI Battle Challenge (Simulation)** — slug
`pokemon-tcg-ai-battle`. Build an AI Training Agent for the cabt Pokémon TCG
engine (built on `kaggle-environments` 1.14.10) and climb the skill-rated
ladder. Companion Hackathon/Strategy track:
`pokemon-tcg-ai-battle-challenge-strategy`. Full rules live in `README.md`
(re-pull with `scripts/kaggle_tool.py pages --save`). Patterned after the
`orbit_wars` repo.

## Environment

Always run Python through `uv` locally (never bare `python`/`python3`):

```bash
uv sync --extra dev
uv run python scripts/kaggle_tool.py leaderboard --top 20
```

Kaggle auth reads `~/.kaggle/kaggle.json`. Data downloads and submissions
require accepting the competition rules on the website first.

## Repository structure

```
src/ptcg_battle/kaggle_client.py  # KaggleClient: leaderboard, submissions, kernels
                                   #   (code), discussions (topics), pages (rules),
                                   #   data files, episodes/replays/logs
scripts/kaggle_tool.py            # one CLI over KaggleClient (subcommands)
scripts/build_submission.py       # package main.py + deck.csv + cg/ -> submission.tar.gz
agent/                            # (your agent) main.py, deck.csv, cg/  — create as needed
outputs/                          # gitignored — caches, leaderboards, replays, kernels
data/                             # gitignored — downloaded card metadata
```

## Agent contract (the one thing to get right)

`main.py` defines `agent(obs_dict) -> list[int]`:

- If `obs.select is None` (initial selection) → return the 60-card deck (list of
  Card IDs).
- Otherwise return **indices into `obs.select.option`**: each in
  `[0, len(option))`, list length in `[minCount, maxCount]`, no duplicates.
- The engine only ever offers legal options; `obs.select.context`
  (`SelectContext`) says what decision is being made.
- Submission bundle = `.tar.gz` with `main.py` at the **root** + `deck.csv` + `cg/`.

## Lint & type-check

```bash
uv run ruff check . && uv run ruff format . && uv run pyright
```

Ruff (`E,F,I,UP,B,SIM`) and Pyright (`basic`) are configured in `pyproject.toml`;
`basic` mode is intentional (kaggle / kaggle-environments ship partial stubs).

## Strategy & RL plan (read before building the agent)

- `rl_research/LESSONS_FROM_ORBIT_WARS.md` — retrospective comparing our Orbit
  Wars attempt (~rank 140, planner fork) against the winner (200M-param pure
  self-play PPO). The core lesson: we mistook a compute/throughput ceiling for an
  algorithmic one and pivoted away from the approach that won. Bet on
  throughput + scale + self-play, not feature engineering or imitation.
- `rl_research/SELFPLAY_RL_PLAN.md` — preliminary, scale-adaptive plan. **Phase 0
  (throughput spike) gates everything** — the cabt engine is a global singleton
  (`Battle.battle_ptr`), so parallelism is process-level only.
- Keep `rl_research/` as the dated experiment log (record dead-ends too).

## Submission limits (don't waste them)

Max **5 submissions/day**; only the latest 2 count for final evaluation. Each new
agent first runs a validation episode vs. itself — if it Errors, pull agent logs
(`scripts/kaggle_tool.py logs <episode_id> 0`).
