"""Observation/action encoder for the self-play RL agent.

Turns a **raw cabt observation dict** (the `obs_dict` handed to `agent()`, i.e.
`Battle.obs` — *not* the parsed `Observation` dataclass) into the flat numeric
arrays the model embeds, plus one **candidate token per legal option** for the
pointer action head.

Why read the raw dict: Phase 0 measured that `to_observation_class()` roughly
*halves* engine throughput (recursive Python `to_dataclass`). The hot training
loop must never pay that — so this module indexes the JSON dict directly. See
`rl_research/PHASE0_THROUGHPUT.md` (finding #2) and `docs/rl-obs-action.md` for
the full contract this code implements.

Design (Lessons 3 & 4): low-level **entity tokens** (active/bench Pokémon, own
hand, board summary) keyed by a learned **card-ID embedding**, plus a learned
token per **legal option**. Legality is free — the engine only ever offers legal
options — so the pointer head scores candidate tokens with no action mask.

The module is intentionally **torch-free** (numpy only) so encoding tests and the
CPU rollout workers don't drag in the heavy RL extra. The model (which *does* use
torch) consumes the arrays defined here; the embedding-table sizes it needs are
exported as the ``*_VOCAB`` / ``*_DIM`` constants below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Vocabulary sizes (index 0 is reserved as PAD / "none" in every table).
#
# Card and attack IDs are dense and 1-based in the shipped engine:
#   all_card_data():  1267 cards, ids 1..1267
#   all_attack():     1556 attacks, ids 1..1556
# We size the embedding tables one past the max id so id N maps to row N and row
# 0 stays free for "no card" / face-down / pad. `tests/test_encoding.py` asserts
# these still bound the live engine, so a mid-competition card drop is caught.
# ---------------------------------------------------------------------------
MAX_CARD_ID = 1267
MAX_ATTACK_ID = 1556
CARD_VOCAB = MAX_CARD_ID + 1  # 1268
ATTACK_VOCAB = MAX_ATTACK_ID + 1  # 1557

N_ENERGY_TYPES = 12  # EnergyType 0..11 (COLORLESS..TEAM_ROCKET)
N_AREA = 13  # AreaType 1..12, row 0 = none
N_OPTION_TYPE = 17  # OptionType 0..16
N_SELECT_CONTEXT = 49  # SelectContext 0..48 (engine may append; see assert in tests)
N_SPECIAL_COND = 6  # 0 = none, 1..5 = POISON..CONFUSE (SpecialConditionType + 1)

# Entity roles (categorical embedding for "what kind of token is this").
ROLE_PAD = 0
ROLE_OWN_ACTIVE = 1
ROLE_OWN_BENCH = 2
ROLE_OPP_ACTIVE = 3
ROLE_OPP_BENCH = 4
ROLE_OWN_HAND = 5
ROLE_GLOBAL = 6  # board-summary token (carries `global_feat`)
ROLE_STADIUM = 7
N_ROLE = 8

# Per-entity numeric feature layout (ENTITY_FEAT_DIM floats, all in ~[0,1]).
#   0 hp/300            5 n_tools/2          10 status: poisoned
#   1 maxHp/300         6 n_pre_evo/2        11 status: burned
#   2 hp/maxHp          7 appearThisTurn     12 status: asleep
#   3 n_energies/6      8 is_active          13 status: paralyzed
#   4 n_energyCards/6   9 is_own             14 status: confused
ENTITY_FEAT_DIM = 15

# Board-summary scalars carried on the GLOBAL token (all from "my" POV, ~[0,1]).
#   0 turn/40            6 my_hand/12          12 energyAttached
#   1 turnAction/20      7 opp_hand/12         13 retreated
#   2 my_prizes/6        8 my_bench/5          14 i_go_first
#   3 opp_prizes/6       9 opp_bench/5         15 stadium_present
#   4 my_deck/60        10 supporterPlayed     16 my_discard/60
#   5 opp_deck/60       11 stadiumPlayed       17 opp_discard/60
GLOBAL_FEAT_DIM = 18

# Per-option numeric extras (OPTION_FEAT_DIM floats).
#   0 number/20   1 count/6   2 has_card   3 has_toolIndex
#   4 has_energyIndex   5 targets_opponent
OPTION_FEAT_DIM = 6

# AreaType ints we resolve cards from (mirror of cg.api.AreaType, kept local so
# this module needs neither the engine nor a chdir to import).
_AREA_DECK = 1
_AREA_HAND = 2
_AREA_DISCARD = 3
_AREA_ACTIVE = 4
_AREA_BENCH = 5
_AREA_PRIZE = 6
_AREA_STADIUM = 7
_AREA_LOOKING = 12

# OptionType ints (mirror of cg.api.OptionType).
_OPT_NUMBER = 0
_OPT_CARD = 3
_OPT_TOOL_CARD = 4
_OPT_ENERGY_CARD = 5
_OPT_ENERGY = 6
_OPT_PLAY = 7
_OPT_ATTACH = 8
_OPT_EVOLVE = 9
_OPT_ABILITY = 10
_OPT_DISCARD = 11
_OPT_ATTACK = 13
_OPT_SKILL = 15
_OPT_SPECIAL_CONDITION = 16


@dataclass
class EncodedObs:
    """Flat numeric encoding of one decision point. Arrays are batch-stackable.

    Entity block (length ``n_entities``): the board as tokens.
    Option block (length ``n_options``): one candidate token per *legal* option,
    index-aligned with ``obs["select"]["option"]`` — candidate ``i`` is the
    pointer-head logit for choosing option ``i`` (this 1:1 alignment is the core
    invariant the round-trip test asserts).
    """

    # --- entity tokens ---
    entity_role: np.ndarray  # int64[n_entities]
    entity_card: np.ndarray  # int64[n_entities]  (card-ID embedding index, 0 = none)
    entity_feat: np.ndarray  # float32[n_entities, ENTITY_FEAT_DIM]
    entity_energy: np.ndarray  # float32[n_entities, N_ENERGY_TYPES] (per-type counts/6)
    # --- option / candidate tokens ---
    opt_type: np.ndarray  # int64[n_options]  (OptionType)
    opt_area: np.ndarray  # int64[n_options]  (AreaType of the referenced card, 0 = none)
    opt_inplay_area: np.ndarray  # int64[n_options]  (AreaType of the in-play target, 0 = none)
    opt_card: np.ndarray  # int64[n_options]  (resolved card-ID of the referent, 0 = none)
    opt_attack: np.ndarray  # int64[n_options]  (attack-ID embedding index, 0 = none)
    opt_special: np.ndarray  # int64[n_options]  (SpecialConditionType + 1, 0 = none)
    opt_feat: np.ndarray  # float32[n_options, OPTION_FEAT_DIM]
    opt_target: np.ndarray  # int64[n_options]  (entity-token index this option acts on, -1 = none)
    # --- globals / decision context ---
    global_feat: np.ndarray  # float32[GLOBAL_FEAT_DIM]
    context: int  # SelectContext of this decision
    min_count: int
    max_count: int
    my_index: int

    @property
    def n_entities(self) -> int:
        return int(self.entity_role.shape[0])

    @property
    def n_options(self) -> int:
        return int(self.opt_type.shape[0])


def _energy_multihot(energies: list[int] | None) -> np.ndarray:
    """Count attached energies per EnergyType, normalised by 6."""
    out = np.zeros(N_ENERGY_TYPES, dtype=np.float32)
    if energies:
        for e in energies:
            if 0 <= e < N_ENERGY_TYPES:
                out[e] += 1.0
    return out / 6.0


def _pokemon_feat(pk: dict, *, is_active: bool, is_own: bool, status: dict) -> np.ndarray:
    """Numeric feature row for one Pokémon entity. `status` holds the active-spot
    special-condition flags (poisoned/burned/...), which apply only to the active."""
    f = np.zeros(ENTITY_FEAT_DIM, dtype=np.float32)
    hp = float(pk.get("hp", 0) or 0)
    max_hp = float(pk.get("maxHp", 0) or 0)
    f[0] = hp / 300.0
    f[1] = max_hp / 300.0
    f[2] = (hp / max_hp) if max_hp > 0 else 0.0
    f[3] = len(pk.get("energies") or []) / 6.0
    f[4] = len(pk.get("energyCards") or []) / 6.0
    f[5] = len(pk.get("tools") or []) / 2.0
    f[6] = len(pk.get("preEvolution") or []) / 2.0
    f[7] = 1.0 if pk.get("appearThisTurn") else 0.0
    f[8] = 1.0 if is_active else 0.0
    f[9] = 1.0 if is_own else 0.0
    if is_active:
        f[10] = 1.0 if status.get("poisoned") else 0.0
        f[11] = 1.0 if status.get("burned") else 0.0
        f[12] = 1.0 if status.get("asleep") else 0.0
        f[13] = 1.0 if status.get("paralyzed") else 0.0
        f[14] = 1.0 if status.get("confused") else 0.0
    return f


def _card_id(card: dict | None) -> int:
    """Card-ID embedding index for a card dict (0 if None / face-down / out of range)."""
    if not card:
        return 0
    cid = card.get("id", 0) or 0
    return cid if 0 < cid <= MAX_CARD_ID else 0


def _resolve_card(obs: dict, area: int | None, index: int | None, player_index: int) -> dict | None:
    """Fetch the card/Pokémon dict referenced by (area, index, playerIndex).

    Raw-dict analogue of `agent/main.py:get_card`. Returns None on anything odd
    (face-down, out of range) so callers degrade to card-ID 0.
    """
    if area is None or index is None:
        return None
    state = obs.get("current")
    if state is None:
        return None
    try:
        if area == _AREA_STADIUM:
            stad = state.get("stadium") or []
            return stad[index] if index < len(stad) else None
        if area == _AREA_LOOKING:
            look = state.get("looking") or []
            return look[index] if index < len(look) else None
        if area == _AREA_DECK:
            deck = (obs.get("select") or {}).get("deck") or []
            return deck[index] if index < len(deck) else None
        players = state.get("players") or []
        if not (0 <= player_index < len(players)):
            return None
        ps = players[player_index]
        zone = {
            _AREA_HAND: ps.get("hand"),
            _AREA_DISCARD: ps.get("discard"),
            _AREA_ACTIVE: ps.get("active"),
            _AREA_BENCH: ps.get("bench"),
            _AREA_PRIZE: ps.get("prize"),
        }.get(area)
        if zone is None:
            return None
        return zone[index] if index < len(zone) else None
    except (IndexError, KeyError, TypeError):
        return None


def encode_observation(obs: dict) -> EncodedObs:
    """Encode one raw observation dict at a decision point (``select`` not None).

    Deck selection (``obs["select"] is None``) is handled outside the model — the
    agent simply returns the 60-card deck — so callers must not pass it here.
    """
    select = obs.get("select")
    state = obs.get("current")
    if select is None or state is None:
        raise ValueError("encode_observation requires a decision obs (select/current not None)")

    my_index = int(state.get("yourIndex", 0))
    op_index = 1 - my_index
    players = state["players"]
    me = players[my_index]
    op = players[op_index]

    # ---- entity tokens, with a position map for option cross-linking ----
    roles: list[int] = []
    cards: list[int] = []
    feats: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    # (player_index, area, index) -> entity token position, for opt_target.
    pos_of: dict[tuple[int, int, int], int] = {}

    def add_pokemon(pk: dict | None, *, role: int, pidx: int, area: int, idx: int, status: dict):
        if pk is None:  # face-down active or empty slot — skip (board summary still counts it)
            return
        pos_of[(pidx, area, idx)] = len(roles)
        roles.append(role)
        cards.append(_card_id(pk))
        feats.append(
            _pokemon_feat(
                pk, is_active=(area == _AREA_ACTIVE), is_own=(pidx == my_index), status=status
            )
        )
        energies.append(_energy_multihot(pk.get("energies")))

    me_status = me
    op_status = op
    me_active = me.get("active") or [None]
    op_active = op.get("active") or [None]
    add_pokemon(
        me_active[0] if me_active else None,
        role=ROLE_OWN_ACTIVE,
        pidx=my_index,
        area=_AREA_ACTIVE,
        idx=0,
        status=me_status,
    )
    for j, pk in enumerate(me.get("bench") or []):
        add_pokemon(
            pk, role=ROLE_OWN_BENCH, pidx=my_index, area=_AREA_BENCH, idx=j, status=me_status
        )
    add_pokemon(
        op_active[0] if op_active else None,
        role=ROLE_OPP_ACTIVE,
        pidx=op_index,
        area=_AREA_ACTIVE,
        idx=0,
        status=op_status,
    )
    for j, pk in enumerate(op.get("bench") or []):
        add_pokemon(
            pk, role=ROLE_OPP_BENCH, pidx=op_index, area=_AREA_BENCH, idx=j, status=op_status
        )

    # Own hand cards (the opponent's hand is hidden — count only).
    for k, card in enumerate(me.get("hand") or []):
        pos_of[(my_index, _AREA_HAND, k)] = len(roles)
        roles.append(ROLE_OWN_HAND)
        cards.append(_card_id(card))
        feats.append(np.zeros(ENTITY_FEAT_DIM, dtype=np.float32))
        energies.append(np.zeros(N_ENERGY_TYPES, dtype=np.float32))

    # Stadium token (shared board zone), if any.
    stadium = state.get("stadium") or []
    if stadium:
        pos_of[(my_index, _AREA_STADIUM, 0)] = len(roles)
        pos_of[(op_index, _AREA_STADIUM, 0)] = len(roles)
        roles.append(ROLE_STADIUM)
        cards.append(_card_id(stadium[0]))
        feats.append(np.zeros(ENTITY_FEAT_DIM, dtype=np.float32))
        energies.append(np.zeros(N_ENERGY_TYPES, dtype=np.float32))

    # Global board-summary token (always last; carries `global_feat`).
    global_token_pos = len(roles)
    roles.append(ROLE_GLOBAL)
    cards.append(0)
    feats.append(np.zeros(ENTITY_FEAT_DIM, dtype=np.float32))
    energies.append(np.zeros(N_ENERGY_TYPES, dtype=np.float32))

    gf = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    gf[0] = state.get("turn", 0) / 40.0
    gf[1] = state.get("turnActionCount", 0) / 20.0
    gf[2] = len(me.get("prize") or []) / 6.0
    gf[3] = len(op.get("prize") or []) / 6.0
    gf[4] = me.get("deckCount", 0) / 60.0
    gf[5] = op.get("deckCount", 0) / 60.0
    gf[6] = me.get("handCount", 0) / 12.0
    gf[7] = op.get("handCount", 0) / 12.0
    gf[8] = len(me.get("bench") or []) / 5.0
    gf[9] = len(op.get("bench") or []) / 5.0
    gf[10] = 1.0 if state.get("supporterPlayed") else 0.0
    gf[11] = 1.0 if state.get("stadiumPlayed") else 0.0
    gf[12] = 1.0 if state.get("energyAttached") else 0.0
    gf[13] = 1.0 if state.get("retreated") else 0.0
    gf[14] = 1.0 if state.get("firstPlayer", -1) == my_index else 0.0
    gf[15] = 1.0 if stadium else 0.0
    gf[16] = len(me.get("discard") or []) / 60.0
    gf[17] = len(op.get("discard") or []) / 60.0

    # ---- option / candidate tokens (1:1 with select.option) ----
    options = select["option"]
    m = len(options)
    opt_type = np.zeros(m, dtype=np.int64)
    opt_area = np.zeros(m, dtype=np.int64)
    opt_inplay = np.zeros(m, dtype=np.int64)
    opt_card = np.zeros(m, dtype=np.int64)
    opt_attack = np.zeros(m, dtype=np.int64)
    opt_special = np.zeros(m, dtype=np.int64)
    opt_feat = np.zeros((m, OPTION_FEAT_DIM), dtype=np.float32)
    opt_target = np.full(m, -1, dtype=np.int64)

    for i, o in enumerate(options):
        t = o.get("type", -1)
        opt_type[i] = t
        area = o.get("area")
        in_area = o.get("inPlayArea")
        opt_area[i] = area or 0
        opt_inplay[i] = in_area or 0
        # playerIndex defaults to the actor (me) when the option omits it.
        p_owner = o.get("playerIndex", my_index)

        # Resolve the referenced card and the in-play target Pokémon token.
        card = None
        if t == _OPT_PLAY:
            card = _resolve_card(obs, _AREA_HAND, o.get("index"), my_index)
        elif t in (_OPT_ATTACH, _OPT_EVOLVE):
            card = _resolve_card(obs, area, o.get("index"), my_index)
        elif t in (_OPT_CARD, _OPT_ABILITY, _OPT_DISCARD):
            card = _resolve_card(obs, area, o.get("index"), p_owner)
        elif t in (_OPT_TOOL_CARD, _OPT_ENERGY_CARD, _OPT_ENERGY):
            # The selectable is a card attached to a Pokémon; resolve the attached
            # card itself for its id, and link the option to the Pokémon token.
            pk = _resolve_card(obs, area, o.get("index"), p_owner)
            if pk is not None:
                if t == _OPT_TOOL_CARD:
                    arr = pk.get("tools") or []
                    ti = o.get("toolIndex", 0) or 0
                    card = arr[ti] if ti < len(arr) else None
                else:
                    arr = pk.get("energyCards") or []
                    ei = o.get("energyIndex", 0) or 0
                    card = arr[ei] if ei < len(arr) else None
        elif t == _OPT_ATTACK:
            opt_attack[i] = o["attackId"] if 0 < (o.get("attackId") or 0) <= MAX_ATTACK_ID else 0
            # Attacks are declared by our active Pokémon.
            opt_target[i] = pos_of.get((my_index, _AREA_ACTIVE, 0), -1)
        elif t == _OPT_SPECIAL_CONDITION:
            sc = o.get("specialConditionType")
            if sc is not None and 0 <= sc < N_SPECIAL_COND - 1:
                opt_special[i] = sc + 1

        opt_card[i] = _card_id(card)

        # Cross-link the option to the entity token it acts on.
        if opt_target[i] < 0:
            if in_area is not None:
                opt_target[i] = pos_of.get((p_owner, in_area, o.get("inPlayIndex", 0)), -1)
            elif (
                t
                in (
                    _OPT_CARD,
                    _OPT_ABILITY,
                    _OPT_DISCARD,
                    _OPT_TOOL_CARD,
                    _OPT_ENERGY_CARD,
                    _OPT_ENERGY,
                )
                and area is not None
            ):
                opt_target[i] = pos_of.get((p_owner, area, o.get("index", -1)), -1)
        # Options with no card/board referent (END, YES/NO, NUMBER, RETREAT) fall
        # back to the global token so the pointer head still has a key to attend.
        if opt_target[i] < 0:
            opt_target[i] = global_token_pos

        # Numeric extras.
        opt_feat[i, 0] = (o.get("number") or 0) / 20.0
        opt_feat[i, 1] = (o.get("count") or 0) / 6.0
        opt_feat[i, 2] = 1.0 if opt_card[i] > 0 else 0.0
        opt_feat[i, 3] = 1.0 if o.get("toolIndex") is not None else 0.0
        opt_feat[i, 4] = 1.0 if o.get("energyIndex") is not None else 0.0
        opt_feat[i, 5] = 1.0 if p_owner == op_index else 0.0

    return EncodedObs(
        entity_role=np.asarray(roles, dtype=np.int64),
        entity_card=np.asarray(cards, dtype=np.int64),
        entity_feat=np.stack(feats).astype(np.float32),
        entity_energy=np.stack(energies).astype(np.float32),
        opt_type=opt_type,
        opt_area=opt_area,
        opt_inplay_area=opt_inplay,
        opt_card=opt_card,
        opt_attack=opt_attack,
        opt_special=opt_special,
        opt_feat=opt_feat,
        opt_target=opt_target,
        global_feat=gf,
        context=int(select.get("context", -1)),
        min_count=int(select.get("minCount", 1)),
        max_count=int(select.get("maxCount", 1)),
        my_index=my_index,
    )
