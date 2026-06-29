"""Deck-agnostic opponent-pool manifest — the league config as DATA (pivot-ready).

A manifest is JSON:

    {"opponents": [
        {"agent": "kaggle:archaludon", "weight": 1.0},
        {"agent": "kaggle:dragapult",  "weight": 1.0},
        {"agent": "heuristic",         "weight": 0.5},
        {"agent": "random",            "weight": 0.3, "deck": "agent/decks/iono.csv"}
    ]}

`agent` is a league spec the worker/collector already understands: ``heuristic`` /
``random`` / ``first`` (stepped locally) or ``kaggle:<name>`` (a vendored module in
`agent/kaggle_agents/`). ``self`` and ``model:<id>`` opponents are added by the
TRAINER (current net + frozen past checkpoints), never by the manifest — they mirror
the trainee deck and carry no deck entry.

Each opponent pilots **its own deck**, resolved here to a `list[int]`:
  * explicit `"deck": "<path>"` wins;
  * else a `kaggle:<name>` spec defaults to `agent/kaggle_agents/<name>_deck.csv`
    (the deck vendored alongside that agent);
  * else `heuristic` defaults to `agent/deck.csv` (main.py's native deck);
  * else (`random`/`first`) → None, i.e. mirror the trainee deck.

Swapping a deck or an opponent agent is a config (manifest) change, not a code
change — that is the whole point. Torch-free so workers/eval can import it cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO / "agent"
KAGGLE_AGENT_DIR = AGENT_DIR / "kaggle_agents"


def read_deck(path: str | Path) -> list[int]:
    """Read a 60-card deck CSV (one Card ID per line) into a list[int]."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    ids = [int(x) for x in p.read_text().split() if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"{p} must have exactly 60 Card IDs, got {len(ids)}")
    return ids


def default_deck_path(spec: str) -> Path | None:
    """The conventional own-deck path for a fixed/Kaggle opponent spec (or None when
    the spec is deck-agnostic and should mirror the trainee deck)."""
    if spec.startswith("kaggle:"):
        return KAGGLE_AGENT_DIR / f"{spec[len('kaggle:') :]}_deck.csv"
    if spec == "heuristic":
        return AGENT_DIR / "deck.csv"
    return None  # random / first → mirror the trainee deck


def resolve_deck(spec: str, deck_field: str | None) -> list[int] | None:
    """Resolve the deck an opponent pilots: explicit field, else the spec default,
    else None (mirror the trainee deck). self/model never reach here."""
    if deck_field:
        return read_deck(deck_field)
    dp = default_deck_path(spec)
    if dp is not None and dp.exists():
        return read_deck(dp)
    return None


@dataclass(frozen=True)
class Opponent:
    spec: str  # league spec: heuristic | random | first | kaggle:<name>
    weight: float
    deck: list[int] | None  # the deck this opponent pilots (None = mirror trainee)


def load_manifest(path: str | Path) -> list[Opponent]:
    """Parse an opponent manifest into resolved `Opponent`s (deck → list[int])."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    data = json.loads(p.read_text())
    out: list[Opponent] = []
    for entry in data.get("opponents", []):
        spec = entry["agent"]
        if spec == "self" or spec.startswith("model:"):
            raise ValueError(f"manifest must not list trainer-managed spec {spec!r}")
        weight = float(entry.get("weight", 1.0))
        deck = resolve_deck(spec, entry.get("deck"))
        out.append(Opponent(spec=spec, weight=weight, deck=deck))
    return out


def manifest_to_league_args(
    opponents: list[Opponent],
) -> tuple[list[tuple[str, float]], dict[str, list[int]]]:
    """Split resolved opponents into `(extra_mix, opp_decks)` for `build_league`."""
    extra = [(o.spec, o.weight) for o in opponents]
    decks = {o.spec: o.deck for o in opponents if o.deck is not None}
    return extra, decks
