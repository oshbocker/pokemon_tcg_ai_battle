"""Frozen card-metadata table tests (torch-free, like the encoding tests).

The table is the `use_card_meta` feature's data half: shape/PAD invariants, the
one-of category partition over the full vocab, range bounds, and spot-checks of
known cards against the raw CSV (`data/EN_Card_Data.csv` — gitignored, so skip
cleanly on a machine that hasn't downloaded it)."""

from __future__ import annotations

import numpy as np
import pytest

from ptcg_battle.card_meta import (
    CARD_META_DIM,
    DEFAULT_CSV,
    FEAT,
    FEATURE_NAMES,
    build_card_meta_table,
)
from ptcg_battle.encoding import CARD_VOCAB, MAX_CARD_ID

pytestmark = pytest.mark.skipif(
    not DEFAULT_CSV.exists(), reason="data/EN_Card_Data.csv not downloaded"
)


@pytest.fixture(scope="module")
def table() -> np.ndarray:
    return build_card_meta_table()


def test_shape_dtype_and_pad_row(table):
    assert table.shape == (CARD_VOCAB, CARD_META_DIM)
    assert table.dtype == np.float32
    assert len(FEATURE_NAMES) == CARD_META_DIM == len(FEAT)
    assert np.all(table[0] == 0.0)  # PAD / "no card" contributes nothing


def test_all_values_bounded_and_every_card_categorized(table):
    assert float(table.min()) >= 0.0 and float(table.max()) <= 1.0
    cats = table[1:, [FEAT["is_pokemon"], FEAT["is_trainer"], FEAT["is_energy"]]]
    assert np.all(cats.sum(axis=1) == 1.0), "every card is exactly one of Pokémon/Trainer/Energy"


def test_spot_check_basic_energy(table):
    v = table[1]  # Card 1 = Basic {G} Energy
    assert v[FEAT["is_energy"]] == 1.0 and v[FEAT["energy_basic"]] == 1.0
    assert v[FEAT["type_G"]] == 1.0
    assert v[FEAT["is_pokemon"]] == 0.0 and v[FEAT["hp_scalar"]] == 0.0


def test_spot_check_supporter(table):
    v = table[1213]  # Card 1213 = Judge (the §4.1 kill-criterion swap card)
    assert v[FEAT["is_trainer"]] == 1.0 and v[FEAT["trainer_supporter"]] == 1.0
    assert v[FEAT["hp_scalar"]] == 0.0 and v[FEAT["retreat"]] == 0.0


def test_spot_check_ex_pokemon(table):
    v = table[190]  # Archaludon ex: Stage 1, 300 HP, {M}, weak {R}, resist {G}, retreat 2
    assert v[FEAT["is_pokemon"]] == 1.0 and v[FEAT["stage_1"]] == 1.0
    assert v[FEAT["is_ex"]] == 1.0 and v[FEAT["is_mega"]] == 0.0
    assert v[FEAT["hp_scalar"]] == pytest.approx(300 / 380)
    assert v[FEAT["hp_bucket_5"]] == 1.0  # 260+ bucket
    assert v[FEAT["type_M"]] == 1.0
    assert v[FEAT["weak_R"]] == 1.0
    assert v[FEAT["resist_G"]] == 1.0
    assert v[FEAT["retreat"]] == pytest.approx(2 / 4)


def test_hp_buckets_are_one_hot_where_hp_known(table):
    buckets = table[:, [FEAT[f"hp_bucket_{i}"] for i in range(6)]]
    assert np.all(buckets.sum(axis=1) <= 1.0)
    # every Pokémon has an HP bucket
    pok = table[:, FEAT["is_pokemon"]] == 1.0
    assert np.all(buckets[pok].sum(axis=1) == 1.0)


def test_vocab_covered(table):
    # ids 1..MAX_CARD_ID all populated (each card is categorized, so nonzero)
    assert int((table[1:].sum(axis=1) > 0).sum()) == MAX_CARD_ID
