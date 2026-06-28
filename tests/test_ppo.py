"""PPO loop tests: act/evaluate_actions consistency (the ratio must be exact at
collection time), GAE math, and rollout/update/eval smoke runs on the engine.

torch is in the optional `rl` extra; engine-touching tests run a couple of real
games (single-process, the engine is a singleton), so they're slower but cheap.
"""

from __future__ import annotations

import math

import pytest

from ptcg_battle.encoding import encode_observation

torch = pytest.importorskip("torch")

from ptcg_battle.model import ModelConfig, PtcgNet, collate  # noqa: E402
from ptcg_battle.ppo import (  # noqa: E402
    PPOConfig,
    adapt_ent_coef,
    collect_rollout,
    gae_terminal,
    load_deck,
    ppo_update,
    quick_eval,
)

TINY = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)


def test_act_matches_evaluate_actions(obs_samples):
    """old_log_prob (from act) must equal evaluate_actions' log-prob at the same
    params — otherwise the PPO ratio is wrong on step 1."""
    encoded = [encode_observation(o) for o in obs_samples]
    torch.manual_seed(0)
    model = PtcgNet(TINY).eval()
    gen = torch.Generator().manual_seed(7)
    acted = model.act(encoded, sample=True, generator=gen)
    actions = [a["action"] for a in acted]
    act_logp = torch.tensor([a["log_prob"] for a in acted])
    with torch.no_grad():
        logp, entropy, value = model.evaluate_actions(collate(encoded), actions)
    assert torch.allclose(logp, act_logp, atol=1e-4)
    assert torch.all(entropy >= -1e-6)
    assert torch.allclose(value, torch.tensor([a["value"] for a in acted]), atol=1e-5)


def test_gae_terminal_undiscounted():
    """γ=λ=1, terminal reward 1, zero values → advantage = return = 1 everywhere."""
    adv, ret = gae_terminal([0.0, 0.0, 0.0], terminal_reward=1.0, gamma=1.0, lam=1.0)
    assert all(abs(a - 1.0) < 1e-9 for a in adv)
    assert all(abs(r - 1.0) < 1e-9 for r in ret)


def test_gae_terminal_discounted_loss():
    """γ<1, λ=1, terminal reward −1, zero values → adv[t] = −γ^(T−1−t); ret = adv."""
    gamma = 0.9
    adv, ret = gae_terminal([0.0, 0.0, 0.0], terminal_reward=-1.0, gamma=gamma, lam=1.0)
    expected = [-(gamma ** (2 - t)) for t in range(3)]
    assert all(math.isclose(a, e, rel_tol=1e-9) for a, e in zip(adv, expected, strict=True))
    assert adv == ret  # zero values → return equals advantage


def test_collect_rollout_smoke():
    """A couple of games vs 'first' yields a finite, consistent buffer."""
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    buf = collect_rollout(model, deck, n_games=2, opponent="first", seed=1)
    assert len(buf) > 0
    assert len(buf.actions) == len(buf) == buf.logp.numel() == buf.adv.numel()
    for t in (buf.logp, buf.value, buf.adv, buf.ret):
        assert torch.all(torch.isfinite(t))
    assert torch.all(buf.logp <= 1e-6)  # log-probs non-positive


def test_collect_selfplay_trains_both_seats():
    """Self-play collects from both seats, so it yields more decisions than a
    one-seat fixed-opponent rollout over the same games."""
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    self_buf = collect_rollout(model, deck, n_games=2, opponent="self", seed=2)
    one_buf = collect_rollout(model, deck, n_games=2, opponent="random", seed=2)
    assert len(self_buf) > len(one_buf)


def test_ppo_update_smoke():
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    buf = collect_rollout(model, deck, n_games=3, opponent="first", seed=3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    m = ppo_update(model, opt, buf, PPOConfig(epochs=2, minibatch=64), device="cpu")
    for k in ("pg_loss", "vf_loss", "entropy", "approx_kl", "clipfrac"):
        assert math.isfinite(m[k])
    assert 0.0 <= m["clipfrac"] <= 1.0
    assert m["entropy"] >= -1e-6


def test_ppo_kl_trust_region_stops_early():
    """A tiny target_kl trips the per-minibatch trust region: the update stops after
    very few minibatches (stopped_kl set), vs a full multi-epoch run when off."""
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    buf = collect_rollout(model, deck, n_games=3, opponent="first", seed=3)
    assert len(buf) > 64  # need several minibatches for the test to be meaningful

    import copy

    m_tight = copy.deepcopy(model)
    opt_t = torch.optim.Adam(m_tight.parameters(), lr=1e-2)  # big steps → KL grows fast
    tight = ppo_update(
        m_tight, opt_t, buf, PPOConfig(epochs=4, minibatch=32, target_kl=1e-6), device="cpu"
    )
    m_off = copy.deepcopy(model)
    opt_o = torch.optim.Adam(m_off.parameters(), lr=1e-2)
    full = ppo_update(
        m_off, opt_o, buf, PPOConfig(epochs=4, minibatch=32, target_kl=0.0), device="cpu"
    )

    assert tight["stopped_kl"] == 1.0
    assert tight["updates"] < full["updates"]  # trust region cut the update short
    assert full["stopped_kl"] == 0.0


def test_adapt_ent_coef_controller():
    """Raises the coef when entropy is below target, lowers it above, no-ops at the
    setpoint, respects the clamp, and is disabled when target<=0."""
    target = 0.05
    assert adapt_ent_coef(0.01, entropy=0.0, target_entropy=target) > 0.01  # too deterministic → up
    assert adapt_ent_coef(0.01, entropy=0.5, target_entropy=target) < 0.01  # too random → down
    # at the setpoint the step is ~1x
    assert abs(adapt_ent_coef(0.01, entropy=target, target_entropy=target) - 0.01) < 1e-9
    # bounded multiplicative step: never more than e^gain per call
    import math

    assert adapt_ent_coef(0.01, 0.0, target, gain=0.3) <= 0.01 * math.exp(0.3) + 1e-9
    # clamps
    assert adapt_ent_coef(0.19, 0.0, target, hi=0.2) <= 0.2
    assert adapt_ent_coef(1.1e-3, 0.5, target, lo=1e-3) >= 1e-3
    # ratchet: lo = initial coef → a high-entropy fresh policy can't cut the bonus below it
    assert adapt_ent_coef(0.02, entropy=0.6, target_entropy=0.1, lo=0.02) == 0.02
    # disabled
    assert adapt_ent_coef(0.01, 0.0, target_entropy=0.0) == 0.01


def test_quick_eval_smoke():
    torch.manual_seed(0)
    model = PtcgNet(TINY)
    deck = load_deck()
    r = quick_eval(model, deck, "random", n_games=4, device="cpu")
    assert r["wins"] + r["losses"] + r["draws"] <= 4
    assert r["n"] == r["wins"] + r["losses"]
    assert r["winrate"] != r["winrate"] or 0.0 <= r["winrate"] <= 1.0  # NaN if no decisive
