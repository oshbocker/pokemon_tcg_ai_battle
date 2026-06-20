# Pokémon TCG AI Battle — Kaggle Agent

Code and tooling for **[The Pokémon Company — PTCG AI Battle Challenge (Simulation)](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)**.
Build an AI Training Agent that plays the Pokémon Trading Card Game, climb the
skill-rated ladder, and (optionally) write it up for the companion
[Hackathon / Strategy track](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy).

> Structured like the `orbit_wars` repo: `uv` for envs, `ruff` + `pyright` for
> lint/type-check, and a Kaggle API helper for pulling discussions/code and
> driving the leaderboard + submissions.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync                 # creates .venv/ and installs from pyproject.toml
uv sync --extra dev     # also installs the kaggle CLI, Jupyter, ruff, pyright
```

Put your Kaggle API token at `~/.kaggle/kaggle.json` (Kaggle → *Account* →
*Create New API Token*), `chmod 600` it, and **accept the competition rules on
the competition page** before downloading data or submitting.

## Kaggle tooling

A single CLI (`scripts/kaggle_tool.py`, backed by `src/ptcg_battle/kaggle_client.py`)
covers the whole loop — discussions, community code, leaderboard, and submissions:

```bash
# Discussions (forum topics) — needs kaggle>=2.2
uv run python scripts/kaggle_tool.py topics --sort top --page-size 20
uv run python scripts/kaggle_tool.py topic 708586          # read one thread

# Community code (notebooks/scripts attached to the competition)
uv run python scripts/kaggle_tool.py kernels --search lucario
uv run python scripts/kaggle_tool.py pull-kernel kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck

# Leaderboard
uv run python scripts/kaggle_tool.py leaderboard --top 20

# Competition rules / overview / evaluation pages (also -> outputs/pages)
uv run python scripts/kaggle_tool.py pages --save

# Card-metadata + sample-submission files
uv run python scripts/kaggle_tool.py files
uv run python scripts/kaggle_tool.py download-data

# Submissions
uv run python scripts/kaggle_tool.py submit submission.tar.gz -m "first agent"
uv run python scripts/kaggle_tool.py submissions
uv run python scripts/kaggle_tool.py episodes <submission_id>
uv run python scripts/kaggle_tool.py replay  <episode_id>
uv run python scripts/kaggle_tool.py logs    <episode_id> 0   # debug an Error
```

## Build a submission

A submission is a `.tar.gz` with `main.py` at the **top level** (not nested),
plus `deck.csv` and the simulator package `cg/`:

```bash
uv run python scripts/build_submission.py \
    --main agent/main.py --deck agent/deck.csv --cg agent/cg
# -> submissions/submission.tar.gz
```

`cg/` ships with the competition's sample submission (`download-data`, then copy
`sample_submission/cg`). Equivalently from a shell: `tar -czvf submission.tar.gz *`.

## Lint & type-check

```bash
uv run ruff check .            # lint
uv run ruff check --fix .      # lint + safe fixes
uv run ruff format .           # format
uv run pyright                 # type-check (src, scripts)
```

---

# Competition rules

> Pulled from the competition's official pages (Description / Evaluation /
> How-to-Play / How-to-Submit / Timeline / Prizes / Data). Authoritative source
> is always the [competition page](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).
> Re-pull anytime with `uv run python scripts/kaggle_tool.py pages --save`.

## Overview

This is the **Simulation** competition (there is a separate
[Strategy/Hackathon track](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy);
entering it is *not* required to compete here).

Pokémon TCG players make decisions while mindful of the opponent's strategy,
deck, and hand, across thousands of card combinations — and with hidden
information, card draws, and coin tosses adding variance. **Not knowing what
cards an opponent holds is a core challenge for an AI Training Agent.** A
simulator (SDK) using the same logic as the Kaggle environment is provided for
local training/testing (suitable for debugging and RL). *Rule-based programming
alone may not ensure a high ranking* — winning requires forward thinking,
real-time adaptation, and optimal decision-making.

## The simulator & agent interface

Battles run on the **cabt Engine**, a Pokémon TCG battle simulator built for
`kaggle-environments` (as of `kaggle-environments` **1.14.10**).

Each turn your agent receives an **observation** — game logs, the current board
state, and a **list of legal options** — and returns the **indices of the
options it selects**. The engine only ever presents legal moves.

- API docs: <https://matsuoinstitute.github.io/cabt/>
- Differences between official Pokémon TCG rules and the simulator are noted in
  [discussion 708586](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708586).

The agent entry point lives in `main.py`:

```python
def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    # Initial selection: obs.select is None -> return your 60-card deck.
    if obs.select is None:
        return my_deck
    # Otherwise return indices into obs.select.option:
    #   * each index in [0, len(obs.select.option))
    #   * list length in [obs.select.minCount, obs.select.maxCount]
    #   * no duplicate indices
    ...
    return chosen_indices
```

`obs.select.context` (a `SelectContext`) tells you what kind of decision is
being asked (e.g. `MAIN`, `SWITCH`, `TO_ACTIVE`, `SETUP_ACTIVE_POKEMON`,
`TO_HAND`, `ATTACH_FROM`). The `cg.api` module exposes `Observation`,
`SelectContext`, `OptionType`, `Card`, `Pokemon`, `all_card_data()`, and
`to_observation_class()`.

## Decks & card data

A deck is exactly **60 Card IDs**, one per line, in `deck.csv`. The competition
provides card metadata to map simulator Card IDs to real cards:

- `Card_ID List_EN.pdf` / `Card_ID List_JP.pdf` — full reference (ID, name,
  expansion, collection no., card image), English and Japanese.
- `EN_Card_Data.csv` / `JP_Card_Data.csv` — structured per-card metadata. Schema:
  Card ID, Card Name, Expansion, Collection No., Stage/Type, Rule, Category
  (Pokémon / Trainer / Energy), Previous stage, HP, Type, Weakness,
  Resistance, Retreat, Move Name, Cost, Damage, Effect Explanation.
- A `sample_submission/` (with `main.py`, `deck.csv`, and the `cg/` package) is
  included to build your first valid bundle.

## How to submit

Submissions are a `.tar.gz` bundle with **`main.py` at the top-level directory
(not nested)** and a **`deck.csv`**. Create it with `tar -czvf submission.tar.gz *`
and upload on the
[My Submissions tab](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submissions),
or submit a notebook's output. Each new submission first plays a **validation
episode against copies of itself** to confirm it works before joining
matchmaking.

## Evaluation & ranking

- **Up to 5 agents per day.** Every submitted agent keeps playing episodes until
  the competition ends; newer agents play more frequently for faster feedback.
  Only the **latest 2 submissions** are tracked for final evaluation.
- On upload, a **validation episode** runs (agent vs. copies of itself). If it
  fails, the submission is marked **Error** (download agent logs to debug);
  otherwise it initializes at **μ₀ = 600** and joins the pool.
- Each submission's **skill rating** is a Gaussian *N(μ, σ²)* — μ is estimated
  skill, σ the uncertainty (shrinks over time). Matchmaking pairs submissions
  with **similar ratings**.
- After each episode, μ moves up for the winner / down for the loser (draws pull
  both μ toward their mean); update magnitude scales with the surprise vs.
  expectation and with σ. **Margin of victory does not affect rating updates.**
- The leaderboard shows only your **best-scoring agent**; track all of them on
  your Submissions page.

## Episode replays / data for BC, IL, RL

- Download replays for your own submissions from the **Submissions** tab or via
  the CLI (`scripts/kaggle_tool.py replay <episode_id>`); other teams' replays
  are available from the **Leaderboard**.
- A **daily export of top-rated episodes** (to help BC/RL/IL) is published in the
  competition forums.
- More on simulation-competition CLI/MCP usage:
  <https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md>

## Timeline

- **June 16, 2026, 11:00 UTC** — Start.
- **Entry deadline** — accept the competition rules before this date to compete.
- **Team merger deadline** — last day to join/merge teams.
- **Final submission deadline** — **Aug 16, 2026** (23:59 UTC).
- **Aug 17 → ~Aug 31, 2026** — games keep running until the ladder converges;
  then the leaderboard is final.

*All deadlines 23:59 UTC unless noted. Organizers may adjust the timeline.*

## Prizes

The **Simulation** competition itself has **no monetary prizes** (Knowledge).
Monetary awards (the challenge advertises up to **$240,000** across the
program) go to the **Hackathon / Strategy track**; final Hackathon rankings
combine Simulation leaderboard performance with the Hackathon evaluation.

## Pokémon TCG resources

- Official rulebook (PDF):
  <https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/meg_rulebook_en.pdf>
- Competition [Data page](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data) — cards & decks.

---

*Pokémon and the Pokémon TCG are trademarks of The Pokémon Company. This is an
independent competition entry, not affiliated with or endorsed by Nintendo /
The Pokémon Company.*
