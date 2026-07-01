"""Frozen static card-metadata table for the `use_card_meta` model feature.

Builds a ``[CARD_VOCAB, CARD_META_DIM]`` float32 table keyed by Card ID from the
competition card dump (`data/EN_Card_Data.csv`), giving every card a fixed,
human-legible representation: category flags, stage / trainer-kind / energy-kind
one-hots, rule-box flags (ex / Mega / ACE SPEC), HP (scalar + bucket one-hot),
retreat cost, and type / weakness / resistance one-hots. All features are in
``[0, 1]``; row 0 (PAD / "no card" / face-down) is all zeros.

Why this exists (coevolution deck search, `rl_research/COEVOLUTIONARY_DECK_SEARCH.md`):
the model's only card knowledge is the *learned* card-ID embedding, so a deck
mutation that introduces a card never seen in training hands the net a ~cold
embedding row — zero-shot play is blind and warm-start fine-tuning must relearn
"what is this card" from outcomes. A frozen metadata feature gives novel cards a
sensible representation for free. `docs/rl-obs-action.md` §0 deliberately left
this out of the default features; it enters the model only behind the
``ModelConfig.use_card_meta`` ablation flag (mirroring ``use_option_rank``).

Torch-free (numpy only), like `encoding.py`: the model converts the array to a
persistent buffer, so shipped checkpoints carry the table and inference (e.g. the
Kaggle bundle, where `data/` is absent) never needs this CSV.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from .encoding import CARD_VOCAB, MAX_CARD_ID

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "EN_Card_Data.csv"

# The Stage/Type column is a closed 9-value vocabulary (asserted at build time).
_STAGES = ["Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"]
_TRAINER_KINDS = ["Item", "Supporter", "Pokémon Tool", "Stadium"]
_ENERGY_KINDS = ["Basic Energy", "Special Energy"]

# Type symbols as printed in the CSV ({G}{R}{W}{L}{P}{F}{D}{M}{C} + 竜 = Dragon).
# Rare unmapped symbols ({A}, {Team Rocket}) fall through to all-zeros.
_TYPES = ["G", "R", "W", "L", "P", "F", "D", "M", "C", "DRAGON"]

# HP bucket edges (upper bound inclusive); pool HP spans 30..380.
_HP_BUCKETS = [60, 90, 130, 180, 250, 10_000]
_HP_SCALE = 380.0
_RETREAT_SCALE = 4.0

# Feature layout — names index the table's second axis via FEAT[name].
FEATURE_NAMES: list[str] = [
    "is_pokemon",
    "is_trainer",
    "is_energy",
    *[f"stage_{i}" for i in range(len(_STAGES))],  # basic / stage1 / stage2
    *[f"trainer_{k.lower().replace(' ', '_').replace('é', 'e')}" for k in _TRAINER_KINDS],
    *[f"energy_{k.split()[0].lower()}" for k in _ENERGY_KINDS],  # basic / special
    "is_ex",
    "is_mega",
    "is_ace_spec",
    "hp_scalar",
    *[f"hp_bucket_{i}" for i in range(len(_HP_BUCKETS))],
    "retreat",
    *[f"type_{t}" for t in _TYPES],
    *[f"weak_{t}" for t in _TYPES],
    *[f"resist_{t}" for t in _TYPES],
]
FEAT: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
CARD_META_DIM = len(FEATURE_NAMES)  # 53

_TYPE_TOKEN = re.compile(r"\{([^}]+)\}")


def _first_type(cell: str) -> str | None:
    """First mapped type symbol in a CSV type cell ('' / 'n/a' / unmapped -> None)."""
    if not cell or cell == "n/a":
        return None
    if "竜" in cell:
        return "DRAGON"
    m = _TYPE_TOKEN.search(cell)
    if m and m.group(1) in _TYPES:
        return m.group(1)
    return None


def _num(cell: str) -> float | None:
    return float(cell) if cell and cell.replace(".", "", 1).isdigit() else None


def build_card_meta_table(csv_path: Path | str | None = None) -> np.ndarray:
    """Build the frozen ``[CARD_VOCAB, CARD_META_DIM]`` float32 metadata table.

    The CSV has one row per (card, move); static fields are identical across a
    card's rows, so the first row per Card ID wins. Raises ``FileNotFoundError``
    if the dump is absent (the model then falls back to a zero table that a
    checkpoint load overwrites)."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"card metadata CSV not found: {path}")

    table = np.zeros((CARD_VOCAB, CARD_META_DIM), np.float32)
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["Card ID"])
            if cid in seen:
                continue  # later rows are extra moves of the same card
            seen.add(cid)
            assert 1 <= cid <= MAX_CARD_ID, f"card id {cid} outside vocab"
            v = table[cid]

            kind = row["Stage (Pokémon)/Type (Energy and Trainer)"]
            if kind in _STAGES:
                v[FEAT["is_pokemon"]] = 1.0
                v[FEAT[f"stage_{_STAGES.index(kind)}"]] = 1.0
            elif kind in _TRAINER_KINDS:
                v[FEAT["is_trainer"]] = 1.0
                v[FEAT[FEATURE_NAMES[6 + _TRAINER_KINDS.index(kind)]]] = 1.0
            elif kind in _ENERGY_KINDS:
                v[FEAT["is_energy"]] = 1.0
                v[FEAT[f"energy_{kind.split()[0].lower()}"]] = 1.0
            else:  # closed vocab — a new engine drop should fail loudly
                raise ValueError(f"unknown Stage/Type {kind!r} for card {cid}")

            rule = row["Rule"]
            if rule.endswith("ex"):  # 'Pokémon ex' | 'Mega Pokémon ex'
                v[FEAT["is_ex"]] = 1.0
                if rule.startswith("Mega"):
                    v[FEAT["is_mega"]] = 1.0
            elif rule == "ACE SPEC":
                v[FEAT["is_ace_spec"]] = 1.0

            hp = _num(row["HP"])
            if hp is not None:
                v[FEAT["hp_scalar"]] = min(hp / _HP_SCALE, 1.0)
                for i, hi in enumerate(_HP_BUCKETS):
                    if hp <= hi:
                        v[FEAT[f"hp_bucket_{i}"]] = 1.0
                        break

            retreat = _num(row["Retreat"])
            if retreat is not None:
                v[FEAT["retreat"]] = min(retreat / _RETREAT_SCALE, 1.0)

            for col, prefix in (
                ("Type", "type"),
                ("Weakness", "weak"),
                ("Resistance (Type)", "resist"),
            ):
                t = _first_type(row[col])
                if t is not None:
                    v[FEAT[f"{prefix}_{t}"]] = 1.0

    assert len(seen) == MAX_CARD_ID, f"expected {MAX_CARD_ID} cards, saw {len(seen)}"
    return table
