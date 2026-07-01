"""Unit tests for the PFSP opponent re-weighting (`train_selfplay.pfsp_weights`).

Pure-logic checks on the priority function: the `var` mode must concentrate weight
on contested (~50%) matchups and floor both solved and hopeless ones; the `hard`
mode must peak on the opponents the learner is losing to; the `wmin` floor must keep
every opponent alive; and an opponent with no win-rate estimate keeps its base weight.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")  # train_selfplay imports torch at module load

_SPEC = importlib.util.spec_from_file_location(
    "train_selfplay", Path(__file__).resolve().parents[1] / "scripts" / "train_selfplay.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_TS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TS)
pfsp_weights = _TS.pfsp_weights


BASE = [("kaggle:dragapult", 1.5), ("kaggle:archaludon", 1.5), ("random", 0.3)]


def _var(ema):
    return dict(pfsp_weights(BASE, ema, mode="var", hard_p=2.0, wmin=0.25, wmax=4.0))


def test_var_peaks_on_contested_matchup():
    """A ~50% opponent gets the max multiplier; a crushed one gets the floor."""
    w = _var({"kaggle:dragapult": 0.5, "kaggle:archaludon": 0.9, "random": 1.0})
    assert w["kaggle:dragapult"] == pytest.approx(1.5 * 4.0)  # 4p(1-p)=1 -> wmax
    assert w["random"] == pytest.approx(0.3 * 0.25)  # p=1 -> wmin floor
    # p=0.9 -> 4*.9*.1=0.36 -> mult 0.25+3.75*0.36=1.6
    assert w["kaggle:archaludon"] == pytest.approx(1.5 * (0.25 + 3.75 * 0.36))


def test_var_floors_hopeless_matchup():
    """An unwinnable opponent (p->0) is floored, not amplified — the key advantage
    over raw inverse-win-rate, which would dump compute into a lost matchup."""
    w = _var({"kaggle:dragapult": 0.0})
    assert w["kaggle:dragapult"] == pytest.approx(1.5 * 0.25)


def test_hard_peaks_on_losing_matchup():
    """`hard` mode amplifies the opponents you're losing to (p->0 -> wmax)."""
    w = dict(
        pfsp_weights(BASE, {"kaggle:dragapult": 0.0}, mode="hard", hard_p=2.0, wmin=0.25, wmax=4.0)
    )
    assert w["kaggle:dragapult"] == pytest.approx(1.5 * 4.0)  # (1-0)^2=1 -> wmax


def test_unseen_opponent_keeps_base_weight():
    """No win-rate estimate yet -> untouched base weight (order preserved)."""
    out = pfsp_weights(BASE, {}, mode="var", hard_p=2.0, wmin=0.25, wmax=4.0)
    assert out == BASE


def test_floor_keeps_every_opponent_alive():
    """Even a fully-solved opponent retains a positive (floored) weight."""
    w = _var({s: 1.0 for s, _ in BASE})
    assert all(v > 0 for v in w.values())
