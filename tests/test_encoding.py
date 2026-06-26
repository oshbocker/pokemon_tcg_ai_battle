"""Round-trip / invariant tests for the obs→tensor + option→candidate encoder.

The headline invariant (the one the obs/action contract is built on): **every
legal option maps to exactly one candidate token**, index-aligned with
``select.option``. Everything else here guards the encoding against silent drift
— ID-vocab bounds, target cross-links staying in range, finite normalised
features, determinism — and independently re-derives a few option referents from
the *dataclass* parse (a different code path than the encoder's raw-dict
indexing) so the two must agree.
"""

from __future__ import annotations

import numpy as np

from ptcg_battle import encoding as E
from ptcg_battle.encoding import encode_observation


def test_one_candidate_token_per_legal_option(obs_samples):
    """1:1 alignment: candidate i is the pointer logit for option i, no more/less."""
    for obs in obs_samples:
        enc = encode_observation(obs)
        options = obs["select"]["option"]
        assert enc.n_options == len(options)
        # The mapping is positional and total: option type is preserved exactly.
        assert enc.opt_type.tolist() == [o["type"] for o in options]


def test_targets_index_valid_entities(obs_samples):
    """Every option's cross-link points at a real entity token (never out of range)."""
    for obs in obs_samples:
        enc = encode_observation(obs)
        assert enc.n_entities >= 1  # global token always present
        if enc.n_options:
            assert enc.opt_target.min() >= 0
            assert enc.opt_target.max() < enc.n_entities


def test_id_vocab_bounds(obs_samples):
    """All card/attack embedding indices stay inside the declared tables."""
    for obs in obs_samples:
        enc = encode_observation(obs)
        assert enc.entity_card.min() >= 0 and enc.entity_card.max() <= E.MAX_CARD_ID
        assert enc.opt_card.min() >= 0 and enc.opt_card.max() <= E.MAX_CARD_ID
        assert enc.opt_attack.min() >= 0 and enc.opt_attack.max() <= E.MAX_ATTACK_ID
        assert enc.opt_special.min() >= 0 and enc.opt_special.max() < E.N_SPECIAL_COND
        assert enc.context < E.N_SELECT_CONTEXT


def test_features_finite_and_normalised(obs_samples):
    """No NaN/Inf, and the normalised feature blocks stay in a sane range."""
    for obs in obs_samples:
        enc = encode_observation(obs)
        for arr in (enc.entity_feat, enc.entity_energy, enc.opt_feat, enc.global_feat):
            assert np.all(np.isfinite(arr))
        # Flags/ratios live in [0,1]; counts are /6 so a hand of 12 energies tops
        # out around 2 — keep a generous ceiling that still catches a unit error.
        assert enc.entity_feat.min() >= 0.0 and enc.entity_feat.max() <= 2.0
        assert enc.global_feat.min() >= 0.0 and enc.global_feat.max() <= 2.0


def test_deterministic(obs_samples):
    """Encoding is pure: same dict in → identical arrays out."""
    for obs in obs_samples:
        a = encode_observation(obs)
        b = encode_observation(obs)
        assert np.array_equal(a.entity_card, b.entity_card)
        assert np.array_equal(a.opt_card, b.opt_card)
        assert np.array_equal(a.opt_target, b.opt_target)
        assert np.allclose(a.entity_feat, b.entity_feat)


def test_play_options_resolve_to_a_hand_card(obs_samples):
    """Independent cross-check via the dataclass parse: every PLAY option's
    resolved card-ID must be a card actually in our hand. Exercises the
    raw-dict resolver against a different (dataclass) indexing path."""
    import sys
    from pathlib import Path

    agent_dir = Path(__file__).resolve().parents[1] / "agent"
    sys.path.insert(0, str(agent_dir))
    from cg.api import OptionType, to_observation_class  # type: ignore[reportMissingImports]

    checked = 0
    for obs in obs_samples:
        oc = to_observation_class(obs)
        if oc.current is None or oc.select is None:
            continue
        me = oc.current.players[oc.current.yourIndex]
        hand_ids = {c.id for c in (me.hand or [])}
        enc = encode_observation(obs)
        for i, o in enumerate(oc.select.option):
            if o.type == OptionType.PLAY:
                assert enc.opt_card[i] in hand_ids, (o.index, enc.opt_card[i], hand_ids)
                checked += 1
    # The mirror games reliably surface PLAY decisions during setup/early turns.
    assert checked > 0, "no PLAY options encountered — fixture coverage regressed"


def test_attack_options_carry_attack_id(obs_samples):
    """ATTACK candidates carry the engine attackId and point at our active token."""
    import sys
    from pathlib import Path

    agent_dir = Path(__file__).resolve().parents[1] / "agent"
    sys.path.insert(0, str(agent_dir))
    from cg.api import OptionType  # type: ignore[reportMissingImports]

    for obs in obs_samples:
        enc = encode_observation(obs)
        for i, o in enumerate(obs["select"]["option"]):
            if o["type"] == OptionType.ATTACK:
                assert enc.opt_attack[i] == o["attackId"]
                assert enc.opt_attack[i] > 0


def test_option_rank_is_clamped_engine_order(obs_samples):
    """opt_rank is the engine option index, clamped to the positional vocab — the
    best->worst prior the (permutation-invariant) pointer head would otherwise lose."""
    for obs in obs_samples:
        enc = encode_observation(obs)
        expected = np.minimum(np.arange(enc.n_options), E.N_OPTION_RANK - 1)
        assert np.array_equal(enc.opt_rank, expected)
        if enc.n_options:
            assert enc.opt_rank.min() >= 0 and enc.opt_rank.max() < E.N_OPTION_RANK


def test_multipick_counts_carried(obs_samples):
    """maxCount>1 decisions preserve the engine's count bounds (never exceeding
    the option list), so the model's multi-pick head has the right constraints."""
    saw_multipick = False
    for obs in obs_samples:
        enc = encode_observation(obs)
        assert 0 <= enc.min_count <= enc.max_count <= enc.n_options
        if enc.max_count > 1:
            saw_multipick = True
    assert saw_multipick, "no multi-pick decision in fixture — coverage regressed"


def test_declared_vocab_bounds_live_engine(engine_card_attack):
    """The hardcoded vocab sizes must still bound the shipped engine — a
    mid-competition card/attack drop trips this before it corrupts embeddings."""
    card_ids, attack_ids = engine_card_attack
    assert max(card_ids) <= E.MAX_CARD_ID, "engine added cards past MAX_CARD_ID"
    assert max(attack_ids) <= E.MAX_ATTACK_ID, "engine added attacks past MAX_ATTACK_ID"
    assert min(card_ids) >= 1 and min(attack_ids) >= 1
