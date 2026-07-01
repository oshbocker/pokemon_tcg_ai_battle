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

## Strategy Category (where the prize money is)

The Simulation ladder is the leaderboard we tune agents against, but the
**Strategy Category** holds the money: **$240k total, 8 finalists × $30k**
(possible Tokyo tournament). Entering it requires competing in Simulation on an
identical team. The deliverable is a **single ≤2000-word Kaggle Writeup**
(hackathon — one submission only). Judged **70% Model / 20% Deck / 10% Report**;
robustness (consistency across matches, low reliance on luck/matchups) is
weighted heavily, and **leaderboard rank alone does not guarantee a good score**
— depth of analysis and reasoning do.

**As we build agents, record strategy notes as we go** in
`rl_research/STRATEGY_WRITEUP_LOG.md` — a living, dated log of decisions,
hypotheses tested, and lessons (including dead-ends). It exists so the final
writeup is a synthesis of real notes, not a scramble. Append to it whenever the
approach or deck strategy shifts.

**Timeline:** Sim final submission **Aug 16 2026** (leaderboard converges
~Aug 31); **Strategy writeup deadline Sept 13 2026**, judging Sept 14 – Oct 11.
No advantage to submitting the writeup early (one-shot, judged after deadline) —
write it against the *final* agent/leaderboard and submit near Sept 13.

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

## The `prepare` gate (run before every commit / submission)

One command — format + lint + type-check + test — mirroring the Orbit Wars
winner's `just prepare` (Lesson 10: scaffold the agentic loop). **Run it and get
a green summary before committing.**

```bash
uv run python scripts/prepare.py          # format-in-place, then lint/type/test
uv run python scripts/prepare.py --check  # verify-only (CI / pre-submit)
# `just prepare` / `just check` are aliases if you have `just` installed.
```

It runs, in order: `ruff format` → `ruff check` (`E,F,I,UP,B,SIM`) → `pyright`
(`basic` mode — kaggle/torch ship partial stubs) → `pytest`. All four are
configured in `pyproject.toml`. Tests live in `tests/` (engine round-trip + eval
math); torch is in the optional `rl` extra (`uv pip install torch ...`; see
pyproject), so the encoding tests stay torch-free.

## RL build artifacts (Phase 1+)

- `src/ptcg_battle/encoding.py` — raw-dict → entity/option tensor encoder (the
  hot-loop encoder; reads the JSON dict, never the dataclass). Contract spec:
  `docs/rl-obs-action.md`; invariants: `tests/test_encoding.py`.
- `src/ptcg_battle/eval_harness.py` + `scripts/eval.py` — high-n side-swapped
  unpaired eval vs a fixed suite (engine is unseedable), resumable CSV, Wilson CIs.
  Eval agents: `heuristic`/`random`/`first`/`mirror` + **`model:<path>`** (a trained
  checkpoint, greedy; torch imported lazily).
- `scripts/bench_inference.py` — P0.3 policy inference-cost probe (needs `rl` extra).
- `src/ptcg_battle/dist_collector.py` (+ torch-free `dist_worker.py`) — P3.1
  distributed self-play collector: a persistent pool of W env-worker processes
  feeding one central batched-GPU inference loop. Drop-in for `ppo.collect_rollout`.
- `scripts/train_selfplay.py` — self-play PPO with `--collector dist`, KL early-stop,
  LR/entropy decay, and `--gate` best-checkpoint promotion vs a frozen last-best.
- `scripts/ablate_option_rank_selfplay.py` — the definitive self-play option-rank A/B.

## Colab artifacts (Google Drive, not git)

Phase 0–3 runs on Colab (L4); `notebooks/colab_selfplay.ipynb` is the Phase-3
notebook. Following the Orbit Wars workflow, **training artifacts persist on Google
Drive, never git** — the notebook mounts Drive and writes every run's
checkpoints/eval-CSVs/logs under `MyDrive/ptcg_outputs/`, so a dropped session
resumes and nothing large goes through the repo (the env binary already lives in
it). Pull results back with rclone:

```bash
# One-time: install rclone + add a 'gdrive' remote
sudo apt install rclone        # or: brew install rclone / curl https://rclone.org/install.sh | sudo bash
rclone config                  # New remote → name "gdrive" → type "drive" → OAuth

# Then fetch what you need
uv run python scripts/download_artifacts.py --list                 # run dirs on Drive
uv run python scripts/download_artifacts.py --logs                 # colab_*.txt → rl_research/ (commit these)
uv run python scripts/download_artifacts.py --run ablation_sp      # A/B ckpts + CSVs → outputs/
```

Commit the small `rl_research/colab_*.txt` logs as the dated experiment record;
checkpoints/CSVs stay gitignored under `outputs/`.

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
