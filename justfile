# Task runner. `just` is optional — every recipe is a thin wrapper over a
# `uv run` command you can also type directly (see CLAUDE.md). Install just with
# `cargo install just` or your package manager if you want the shortcuts.

# The single quality gate (format + lint + type-check + test). Run before commits.
prepare:
    uv run python scripts/prepare.py

# Verify-only variant for CI / pre-submission (does not rewrite files).
check:
    uv run python scripts/prepare.py --check

# Lint + format-check only (fast).
lint:
    uv run ruff check . && uv run ruff format --check .

# Tests only.
test:
    uv run pytest

# Local self-play validation (mirrors Kaggle's validation episode).
selfplay:
    uv run python scripts/local_selfplay.py

# High-n side-swapped eval vs the honest suite (resumable).
eval *ARGS:
    uv run python scripts/eval.py {{ARGS}}
