"""Phase 2 model tests: shapes, pointer masking, valid action sampling, the
option-rank ablation toggle, and an overfit-one-batch capacity probe.

The overfit probe mirrors the Kaggle community's finding (#713608) that the value
head's mid-training weakness was *data/measurement*, not capacity: on a single
fixed batch the pointer must be able to drive the cross-entropy to ~0. If this
ever fails, the bug is in the model wiring, not the RL loop.

torch lives in the optional `rl` extra, so skip cleanly if it isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ptcg_battle.encoding import encode_observation

torch = pytest.importorskip("torch")

from ptcg_battle.model import (  # noqa: E402
    SIZE_BANDS,
    ModelConfig,
    PtcgNet,
    collate,
    param_counts,
    synthetic_collated,
)


@pytest.fixture(scope="module")
def encoded(obs_samples):
    return [encode_observation(o) for o in obs_samples]


def test_forward_shapes_and_value_range(encoded):
    torch.manual_seed(0)
    model = PtcgNet(SIZE_BANDS["tiny"]).eval()
    batch = collate(encoded)
    with torch.no_grad():
        logits, value = model(batch)
    b = len(encoded)
    assert logits.shape == (b, batch["opt_mask"].shape[1])
    assert value.shape == (b,)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)  # tanh critic


def test_padded_options_are_masked(encoded):
    torch.manual_seed(0)
    model = PtcgNet(SIZE_BANDS["tiny"]).eval()
    batch = collate(encoded)
    with torch.no_grad():
        logits, _ = model(batch)
    pad = ~batch["opt_mask"]
    if pad.any():
        assert torch.all(logits[pad] < -1e8)  # padded candidates can never be sampled


def test_small_band_param_count(encoded):
    """The first training band must actually be ~5.6M (the P0.3 `small` decision)."""
    model = PtcgNet(SIZE_BANDS["small"])
    total, nonemb = param_counts(model)
    assert 5.0e6 < total < 6.5e6
    assert nonemb < total  # embeddings are a real chunk


def test_act_returns_valid_selections(encoded):
    torch.manual_seed(0)
    model = PtcgNet(SIZE_BANDS["tiny"]).eval()
    gen = torch.Generator().manual_seed(0)
    res = model.act(encoded, sample=True, generator=gen)
    assert len(res) == len(encoded)
    saw_multipick = False
    for e, r in zip(encoded, res, strict=True):
        a = r["action"]
        assert len(set(a)) == len(a)  # distinct
        assert all(0 <= i < e.n_options for i in a)  # in range
        assert e.min_count <= len(a) <= e.max_count  # honors engine count bounds
        assert -1.0 <= r["value"] <= 1.0
        assert r["log_prob"] <= 1e-6  # log prob is non-positive
        if e.max_count > 1:
            saw_multipick = True
    assert saw_multipick, "fixture lost multi-pick coverage"


def test_act_greedy_is_deterministic(encoded):
    torch.manual_seed(0)
    model = PtcgNet(SIZE_BANDS["tiny"]).eval()
    a = model.act(encoded, sample=False)
    b = model.act(encoded, sample=False)
    assert [x["action"] for x in a] == [x["action"] for x in b]


def test_option_rank_toggle_changes_outputs(encoded):
    """The ablation lever must actually do something: identical init, rank on vs off,
    different logits — and off-mode must not consult the rank embedding."""
    batch = collate(encoded)
    torch.manual_seed(0)
    on = PtcgNet(ModelConfig(d_model=128, n_layers=2, n_heads=4, d_ff=256, use_option_rank=True))
    torch.manual_seed(0)
    off = PtcgNet(ModelConfig(d_model=128, n_layers=2, n_heads=4, d_ff=256, use_option_rank=False))
    with torch.no_grad():
        lon, _ = on.eval()(batch)
        loff, _ = off.eval()(batch)
    valid = batch["opt_mask"]
    assert float((lon[valid] - loff[valid]).abs().max()) > 1e-3


def test_card_meta_warm_parity_and_toggle(encoded):
    """`use_card_meta` contract: (a) at init a meta-ON net is behavior-IDENTICAL to
    the meta-OFF net (zero-init projection, created last so shared modules draw the
    same RNG) — that's what makes grafting it onto a meta-OFF parent via
    `--init-ckpt` behavior-preserving; (b) the buffer has the right shape with a
    zero PAD row and ships in the state_dict; (c) once the projection is nonzero
    the feature actually reaches the logits."""
    from ptcg_battle.card_meta import CARD_META_DIM, DEFAULT_CSV
    from ptcg_battle.encoding import CARD_VOCAB

    if not DEFAULT_CSV.exists():
        pytest.skip("data/EN_Card_Data.csv not downloaded")
    batch = collate(encoded)
    torch.manual_seed(0)
    on = PtcgNet(ModelConfig(d_model=128, n_layers=2, n_heads=4, d_ff=256, use_card_meta=True))
    torch.manual_seed(0)
    off = PtcgNet(ModelConfig(d_model=128, n_layers=2, n_heads=4, d_ff=256, use_card_meta=False))
    with torch.no_grad():
        lon, _ = on.eval()(batch)
        loff, _ = off.eval()(batch)
    valid = batch["opt_mask"]
    assert float((lon[valid] - loff[valid]).abs().max()) < 1e-6  # warm-start parity

    assert on.card_meta_table.shape == (CARD_VOCAB, CARD_META_DIM)
    assert torch.all(on.card_meta_table[0] == 0)  # PAD row
    assert "card_meta_table" in on.state_dict()  # persists into checkpoints
    assert "card_meta_table" not in off.state_dict()

    with torch.no_grad():
        on.card_meta_proj.weight.normal_(0.0, 0.1)
        lon2, _ = on(batch)
    assert float((lon2[valid] - loff[valid]).abs().max()) > 1e-3


def test_overfit_one_batch(encoded):
    """Capacity probe: the pointer must drive a fixed single-pick batch's CE to ~0.

    Use only single-pick decisions and a fixed pseudo-random target per sample; if
    the head can't memorize one batch, the model wiring is broken (not the RL)."""
    single = [e for e in encoded if e.max_count == 1][:8]
    assert len(single) >= 4
    batch = collate(single)
    # Deterministic target in each sample's legal range.
    counts = batch["opt_mask"].sum(1)
    targets = torch.tensor([int(0 if c <= 1 else (i % int(c))) for i, c in enumerate(counts)])

    torch.manual_seed(0)
    model = PtcgNet(ModelConfig(d_model=128, n_layers=2, n_heads=4, d_ff=256)).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss0 = None
    for step in range(250):
        logits, _ = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if step == 0:
            loss0 = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = float(loss.detach())
    assert final < 0.05, f"could not overfit one batch: {loss0:.3f} -> {final:.3f}"
    with torch.no_grad():
        pred = model(batch)[0].argmax(1)
    assert torch.equal(pred, targets)


def test_synthetic_collated_matches_model(encoded):
    """The probe's synthetic batch must feed the real model (no shape drift)."""
    torch.manual_seed(0)
    model = PtcgNet(SIZE_BANDS["tiny"]).eval()
    batch = synthetic_collated(b=4, n_ent=16, n_opt=20)
    with torch.no_grad():
        logits, value = model(batch)
    assert logits.shape == (4, 20)
    assert value.shape == (4,)
    assert np.all(np.isfinite(value.numpy()))
