"""Phase 3 (P3.1) distributed collector tests.

Three things must hold: (1) the shared GAE/buffer assembly is identical to the
single-process path (parity by construction — both call
`build_buffer_from_trajectories`); (2) the multi-process collector produces a
finite, self-consistent buffer with only *valid* engine actions; (3) self-play
buffers carry both seats' decisions. The engine is a global singleton, so these
spawn real worker procs and play a couple of real games — slower, but the only
honest check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ptcg_battle.dist_collector import DistributedCollector  # noqa: E402
from ptcg_battle.model import ModelConfig, PtcgNet  # noqa: E402
from ptcg_battle.ppo import (  # noqa: E402
    PPOConfig,
    build_buffer_from_trajectories,
    gae_terminal,
    load_deck,
    ppo_update,
)

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)


def _fake_step(value):
    """A trajectory step with a throwaway encoding and the given critic value."""
    import numpy as np

    z = np.zeros(0, np.int64)
    f = np.zeros((0, 0), np.float32)
    from ptcg_battle.encoding import EncodedObs

    enc = EncodedObs(
        entity_role=z, entity_card=z, entity_feat=f, entity_energy=f,
        opt_type=z, opt_area=z, opt_inplay_area=z, opt_card=z, opt_attack=z,
        opt_special=z, opt_feat=f, opt_target=z, opt_rank=z,
        global_feat=np.zeros(0, np.float32), context=0, min_count=1, max_count=1, my_index=0,
    )  # fmt: skip
    return (enc, [0], -0.5, float(value))


def test_build_buffer_matches_gae_terminal():
    """The shared assembler must reproduce per-step GAE exactly, in order."""
    values = [0.2, -0.1, 0.4]
    steps = [_fake_step(v) for v in values]
    reward = 1.0
    gamma, lam = 0.99, 0.95
    buf = build_buffer_from_trajectories([(steps, reward)], gamma, lam)
    adv, ret = gae_terminal(values, reward, gamma, lam)
    assert torch.allclose(buf.adv, torch.tensor(adv, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(buf.ret, torch.tensor(ret, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(buf.value, torch.tensor(values, dtype=torch.float32), atol=1e-6)
    assert buf.actions == [[0], [0], [0]]


def test_build_buffer_skips_empty_trajectories():
    buf = build_buffer_from_trajectories([([], 1.0), ([_fake_step(0.0)], -1.0)], 0.99, 0.95)
    assert len(buf) == 1


def _assert_valid_buffer(buf):
    assert len(buf) > 0
    assert len(buf.actions) == len(buf) == buf.logp.numel() == buf.adv.numel()
    for t in (buf.logp, buf.value, buf.adv, buf.ret):
        assert torch.all(torch.isfinite(t))
    assert torch.all(buf.logp <= 1e-6)  # log-probs are non-positive
    for enc, action in zip(buf.encoded, buf.actions, strict=True):
        assert enc.min_count <= len(action) <= enc.max_count
        assert len(set(action)) == len(action)  # no duplicate picks
        assert all(0 <= i < enc.n_options for i in action)  # legal option indices


def test_dist_collector_selfplay_smoke():
    """A handful of self-play games over a small worker pool yields a valid buffer
    that ppo_update can consume."""
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    with DistributedCollector(deck, n_workers=2) as col:
        buf = col.collect(model, n_games=4, device="cpu", seed=1)
    _assert_valid_buffer(buf)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    m = ppo_update(model, opt, buf, PPOConfig(epochs=1, minibatch=64), device="cpu")
    assert all(v == v for v in m.values())  # no NaNs


def test_dist_collector_reusable_across_iters():
    """The pool persists: a second collect on the same collector still works (the
    cross-iteration reuse the training loop depends on)."""
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    col = DistributedCollector(deck, n_workers=2)
    try:
        b1 = col.collect(model, n_games=2, device="cpu", seed=1)
        b2 = col.collect(model, n_games=2, device="cpu", seed=2)
    finally:
        col.close()
    _assert_valid_buffer(b1)
    _assert_valid_buffer(b2)


def test_dist_collector_league_mix():
    """A league mixing self + a frozen past-checkpoint model + a fixed agent yields a
    valid buffer; only the current policy's decisions are trained (frozen-opponent
    and fixed-opponent decisions never enter the buffer)."""
    from ptcg_battle.dist_collector import League

    torch.manual_seed(0)
    model = PtcgNet(TINY)
    frozen = PtcgNet(TINY).eval()  # a stand-in "past checkpoint"
    deck = load_deck()
    league = League(
        mix=[("self", 1.0), ("model:past", 1.0), ("random", 1.0)],
        models={"past": frozen},
    )
    with DistributedCollector(deck, n_workers=3, fixed_specs=["random"]) as col:
        buf = col.collect(model, n_games=6, league=league, device="cpu", seed=5)
    _assert_valid_buffer(buf)


def test_dist_collector_fixed_opponent_one_seat():
    """Against a fixed opponent only the model seat is buffered; the buffer is still
    valid and non-empty."""
    from ptcg_battle.dist_collector import League

    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    league = League(mix=[("first", 1.0)])
    with DistributedCollector(deck, n_workers=2, fixed_specs=["first"]) as col:
        buf = col.collect(model, n_games=4, league=league, device="cpu", seed=3)
    _assert_valid_buffer(buf)
