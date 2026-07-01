"""model:<path> eval wiring (P3 task 2).

A trained checkpoint must be loadable as an evaluation champion through the same
high-n, side-swapped harness everything else uses. We save a tiny net, then (a)
check the resolved agent emits a *legal* selection on real obs and (b) run a few
real games through `evaluate()` end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ptcg_battle.eval_harness import _build_model_agent, evaluate  # noqa: E402
from ptcg_battle.model import ModelConfig, PtcgNet  # noqa: E402

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)


def _save_tiny_ckpt(path: Path) -> None:
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    torch.save({"model": model.state_dict(), "cfg": TINY.__dict__, "iter": 0}, path)


def test_model_agent_emits_legal_selection(tmp_path, obs_samples):
    ckpt = tmp_path / "tiny.pt"
    _save_tiny_ckpt(ckpt)
    agent_fn, deck = _build_model_agent(f"model:{ckpt}")
    assert len(deck) == 60
    for obs in obs_samples:
        sel = obs["select"]
        action = agent_fn(obs)
        n = len(sel["option"])
        assert sel["minCount"] <= len(action) <= sel["maxCount"]
        assert len(set(action)) == len(action)
        assert all(0 <= i < n for i in action)


def test_model_agent_loads_card_meta_ckpt(tmp_path, obs_samples):
    """A `use_card_meta` checkpoint round-trips through the eval harness unchanged:
    the net is rebuilt from the ckpt's cfg (which carries the flag) and the frozen
    metadata table rides along as a persistent buffer in the state_dict."""
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, use_card_meta=True)
    torch.manual_seed(0)
    model = PtcgNet(cfg)
    ckpt = tmp_path / "meta.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "iter": 0}, ckpt)
    agent_fn, deck = _build_model_agent(f"model:{ckpt}")
    assert len(deck) == 60
    obs = obs_samples[0]
    action = agent_fn(obs)
    sel = obs["select"]
    assert sel["minCount"] <= len(action) <= sel["maxCount"]
    assert all(0 <= i < len(sel["option"]) for i in action)


def test_model_champion_runs_through_evaluate(tmp_path):
    ckpt = tmp_path / "tiny.pt"
    _save_tiny_ckpt(ckpt)
    out = tmp_path / "eval.csv"
    summaries = evaluate(
        champion=f"model:{ckpt}",
        opponents=["random"],
        games=4,
        out_csv=out,
        workers=2,
        chunk=1,
    )
    s = summaries["random"]
    assert s.games == 4
    assert s.wins + s.losses + s.draws + s.errors == 4
    assert out.exists()
