import os
from collections import defaultdict

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    to_observation_class,
)

# --- Optional forward-search API (engine exposes a forward model). ----------
# We import defensively: if the names differ in the deployed SDK, USE_SEARCH
# simply stays disabled and we fall back to the heuristic. Nothing crashes.
_SEARCH_OK = False
try:
    from cg.api import search_begin, search_release, search_step  # type: ignore

    _SEARCH_OK = True
except Exception:
    _SEARCH_OK = False

# ============================================================================
# CONFIG
# ============================================================================
# Keep this False for your first safe submissions. It ships the proven
# heuristic, hardened against crashes. Turn it on ONLY after you have verified
# locally (see local_harness.py -> smoke_test_search) that search_begin works
# in your SDK build and stays within the per-turn time budget. The agent always
# falls back to the heuristic on any search error, so it can never crash even
# if you enable it before fully validating.
USE_SEARCH = False
SEARCH_TIME_BUDGET = 1.5  # seconds, soft cap per decision when searching
SEARCH_MAX_CANDIDATES = 6  # how many first-actions to roll out

# ============================================================================
# DECK
# ============================================================================
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path) as f:
    _csv = f.read().split("\n")
my_deck = [int(_csv[i]) for i in range(60)]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist IDs (used by the rule-based policy)
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6

# Deck-out guard: below this many cards left, stop firing draw-heavy actions
# (Carmine / Lillie / Lunatone's draw ability) so we don't deck ourselves out.
LOW_DECK_COUNT = 8

# --- Meta tech: the Day-1 #1 deck is a Crustle wall. -----------------------
# Crustle (345) ability "Mysterious Rock Inn" negates ALL damage from the
# opponent's Pokemon ex. Mega Lucario ex is a mega-ex, so swinging it into
# Crustle does ZERO damage. The deck's answer is Hariyama (non-ex, 210): the
# ability does not stop non-ex attackers. The policy below routes around the
# wall instead of whiffing ex attacks into it.
Crustle = 345
CRUSTLE_AWARE = True  # set False to reproduce the old "ex into the wall" behavior


class AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False


plan = AttackPlan()
pre_turn = 0
ability_used = False


# ============================================================================
# HELPERS (from the proven sample agent)
# ============================================================================
def get_card(obs, area, index, player_index):
    """Safely pull a Card/Pokemon from a zone. Returns None on anything odd."""
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


def prize_count(pokemon):
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie's Pearl
            count -= 1
    return max(0, count)


def pokemon_score(pokemon):
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    pid = pokemon.id
    # De-prioritise low-value supports (Squawkabilly ex, Noctowl, Fan Rotom, Archaludon ex)
    if pid == 144 or pid == 322 or pid == 323 or pid == 337:
        score -= 200
    if pid == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


# ============================================================================
# HEURISTIC POLICY  (organizers' tuned logic, kept intact)
# ============================================================================
def heuristic_agent(obs):
    """Returns the option indices for the current selection (descending score)."""
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)
    # deckCount may be absent in odd states; default high so the guard is off.
    low_deck = getattr(my_state, "deckCount", 999) <= LOW_DECK_COUNT

    global plan, pre_turn, ability_used
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    attacker1 = False
    attacker2 = False
    for card in my_state.active + my_state.bench:
        if card is None:
            continue
        field_counts[card.id] += 1
        if card.id == Makuhita or card.id == Hariyama:
            if len(card.energies) >= 3:
                attacker2 = True
        elif card.id == Riolu or card.id == Mega_Lucario_ex:
            if len(card.energies) >= 2:
                attacker1 = True

    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        can_use_mega_brave = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card and card.id == Switch:
                    can_switch = True
                elif card and card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.EVOLVE:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card and card.id == Hariyama:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True
                if o.attackId == 983:  # Mega Brave
                    can_use_mega_brave = True

        my_cards = [my_state.active[0]] + list(my_state.bench)
        op_cards = [op_state.active[0]] + list(op_state.bench)

        if state.turn >= 2:
            best_score = -1
            for i, my_pokemon in enumerate(my_cards):
                if my_pokemon is None:
                    continue
                if i != 0 and not can_switch:
                    break
                for a in range(2):
                    energy_required = 0
                    base_damage = 0
                    base_score = 0
                    if my_pokemon.id == Mega_Lucario_ex:
                        if a == 0:
                            energy_required = 1
                            base_damage = 130
                            base_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                        else:
                            energy_required = 2
                            base_damage = 270
                        if my_prize == 2 or my_prize == 3:
                            base_score -= 500
                    elif a == 1:
                        break
                    elif my_pokemon.id == Hariyama:
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Makuhita:
                        for o in select.option:
                            if o.type == OptionType.EVOLVE:
                                index = o.inPlayIndex
                                if o.inPlayArea == AreaType.BENCH:
                                    index += 1
                                if index == i:
                                    break
                        else:
                            break
                        base_score -= 100
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Solrock:
                        if field_counts[Lunatone] >= 1:
                            energy_required = 1
                            base_damage = 70

                    if base_damage <= 0:
                        continue

                    more_energy = False
                    energy_count = len(my_pokemon.energies)
                    if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                        break
                    if energy_count < energy_required:
                        if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                            energy_count += 1
                            if energy_count < energy_required:
                                continue
                            else:
                                more_energy = True
                        else:
                            continue

                    for j, op_pokemon in enumerate(op_cards):
                        if op_pokemon is None:
                            continue
                        if j != 0 and not can_op_switch:
                            break
                        damage = base_damage
                        data = card_table[op_pokemon.id]
                        if data.weakness == EnergyType.FIGHTING:
                            damage *= 2
                        elif data.resistance == EnergyType.FIGHTING:
                            damage -= 30
                        # Crustle wall: its ability voids all damage from our
                        # ex / mega-ex attackers. Treat that attack as a whiff
                        # so a non-ex attacker (Hariyama) or a Boss'd bench
                        # target is preferred instead.
                        my_data = card_table[my_pokemon.id]
                        crustle_immune = (
                            CRUSTLE_AWARE
                            and op_pokemon.id == Crustle
                            and (my_data.ex or my_data.megaEx)
                        )
                        if crustle_immune:
                            damage = 0
                        prize = 0
                        score = pokemon_score(op_pokemon)
                        if op_pokemon.hp <= damage:
                            prize = prize_count(op_pokemon)
                        else:
                            score *= damage / op_pokemon.hp
                        score += base_score

                        if len(op_state.prize) <= prize:
                            score = 50000

                        if crustle_immune:
                            # Never choose a guaranteed-zero swing into the wall.
                            score = -10000

                        if i == 0:
                            score += 220
                        if j == 0:
                            score += 300
                        score += energy_count
                        if best_score < score:
                            best_score = score
                            plan.attacker = i
                            plan.target = j
                            plan.attack_index = a
                            plan.remain_hp = op_pokemon.hp - damage
                            plan.energy = more_energy

    def energy_score(pokemon, active):
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Makuhita or pokemon.id == Hariyama:
            if pokemon.id == Hariyama:
                score += 1
            if energy_count < 3:
                score += 100
            if attacker2:
                score -= 50
        elif pokemon.id == Lunatone:
            score -= 100
        elif pokemon.id == Solrock:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id == Riolu or pokemon.id == Mega_Lucario_ex:
            if pokemon.id == Mega_Lucario_ex:
                score += 1
            if energy_count < 2:
                score += 100
            if attacker1:
                score -= 50
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += 100
                        if card.id == Mega_Lucario_ex:
                            score += 8 if (my_prize == 2 or my_prize == 3) else 20
                        elif card.id == Hariyama and energy_count >= 2:
                            score += 15
                        elif card.id == Makuhita and energy_count >= 2:
                            score += 10
                        elif card.id == Solrock:
                            score += 5
                        elif card.id == Riolu:
                            score += 4
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Solrock:
                        score = 2 if state.firstPlayer == my_index else 4
                    elif card.id == Riolu:
                        score = 3
                    elif card.id == Makuhita:
                        score = 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Makuhita:
                        score += -10 if field_counts[card.id] >= 1 else 10
                    elif card.id == Hariyama:
                        score += 20 if field_counts[Makuhita] >= 1 else -20
                    elif card.id == Lunatone:
                        score += -250 if field_counts[card.id] >= 1 else 60
                    elif card.id == Solrock:
                        score += -250 if field_counts[card.id] >= 1 else 50
                    elif card.id == Riolu:
                        if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                            score -= 150
                        elif field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 1:
                            score -= 3
                        else:
                            score += 40
                    elif card.id == Mega_Lucario_ex:
                        score += 40 if field_counts[Riolu] >= 1 else -15
                    elif card.id == Basic_Fighting_Energy:
                        score += 30 if (not ability_used or not state.energyAttached) else -1
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif (
                    context == SelectContext.SETUP_BENCH_POKEMON
                    or context == SelectContext.TO_BENCH
                ):
                    # Bench the Lucario line (Riolu) first, then the draw engine.
                    data = card_table.get(card.id)
                    if data is not None and data.cardType == CardType.POKEMON:
                        if card.id == Riolu:
                            score = 120 - 25 * field_counts[Riolu]
                        elif card.id == Solrock:
                            score = 90 if field_counts[Solrock] == 0 else -1
                        elif card.id == Lunatone:
                            score = 80 if field_counts[Lunatone] == 0 else -1
                        elif card.id == Makuhita:
                            score = 65 if field_counts[Makuhita] == 0 else 10
                elif context == SelectContext.DISCARD:
                    # Pitch redundant/dead cards; protect key pieces.
                    cid = card.id
                    if cid == Basic_Fighting_Energy:
                        score = 45 if hand_counts[cid] >= 2 else 5
                        if plan.energy and not state.energyAttached:
                            score -= 200
                    elif hand_counts[cid] >= 2:
                        score = 70
                    elif (cid == Lunatone or cid == Solrock) and field_counts[cid] >= 1:
                        score = 55
                    elif cid == Gravity_Mountain and stadium_id == Gravity_Mountain:
                        score = 50
                    elif (cid == Carmine or cid == Lillie_Determination) and state.supporterPlayed:
                        score = 30
                    elif cid == Mega_Lucario_ex and field_counts[Riolu] == 0:
                        score = -80
                    elif cid == Hariyama and field_counts[Makuhita] == 0:
                        score = -50
                    elif cid in (Riolu, Makuhita, Boss_Orders, Hero_Cape):
                        score = -40
                elif (
                    context == SelectContext.DAMAGE_COUNTER
                    or context == SelectContext.DAMAGE_COUNTER_ANY
                ):
                    if isinstance(card, Pokemon):
                        if o.playerIndex != my_index:
                            score = 10000 + prize_count(card) * 1000 - getattr(card, "hp", 0)
                        else:
                            score = -pokemon_score(card)
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is None:
                scores.append(0)
                continue
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Lunatone or card.id == Solrock:
                    if field_counts[card.id] >= 1:
                        score = -1
                elif card.id == Riolu:
                    if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                        score = -1
            else:
                score = 10000
                if card.id == Switch:
                    score = -1 if plan.attacker <= 0 else 6000
                elif card.id == Premium_Power_Pro:
                    if state.supporterPlayed and plan.remain_hp <= 0:
                        score = -1
                    elif not can_attack:
                        if (
                            not state.supporterPlayed
                            and hand_counts[Carmine] > 0
                            and hand_counts[Lillie_Determination] == 0
                        ):
                            score = 3050
                        else:
                            score = -1
                    else:
                        score = 5000
                elif card.id == Boss_Orders:
                    score = 3200 if plan.target >= 1 else -1
                elif card.id == Carmine:
                    score = -1 if low_deck else 3000
                elif card.id == Lillie_Determination:
                    score = -1 if low_deck else 3100
                elif card.id == Gravity_Mountain:
                    if stadium_id == 0:
                        score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is None or pokemon is None:
                scores.append(0)
                continue
            if card.id == Hero_Cape:
                score = 7000
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if o.inPlayArea == AreaType.ACTIVE:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                else:
                    if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if pokemon is None:
                scores.append(0)
                continue
            score = 9000 + len(pokemon.energies)
            if pokemon.id == Makuhita and plan.target == 0:
                score = -1
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == 1267:  # Lumiose City
                score = 1
            elif card is not None and card.id == Lunatone and low_deck:
                score = -1  # Lunar Cycle draws 3 -> don't deck ourselves out
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            score = 2000 if plan.attacker >= 1 else -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if plan.attack_index == 1:
                if o.attackId == 983:
                    score += 100
            else:
                if o.attackId != 983:
                    score += 100
        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    if context == SelectContext.MAIN:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == Lunatone:
                ability_used = True
    return desc_indices


# ============================================================================
# STATE EVALUATION  (used only by the optional forward search)
# ============================================================================
def evaluate_state(obs):
    """Heuristic value of a board from our perspective. Higher = better."""
    st = obs.current
    if st is None:
        return 0.0
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]

    # Terminal: prizes are the win condition (6 -> 0).
    val = 0.0
    val += (len(op.prize) - len(me.prize)) * 10000.0  # prize race dominates

    # Reward having set-up attackers and energy in play.
    for p in [me.active[0] if me.active else None] + list(me.bench):
        if p is None:
            continue
        val += len(p.energies) * 120.0
        if p.id == Mega_Lucario_ex:
            val += 400.0
        if p.id == Hariyama:
            val += 200.0
    # Penalise our active being low / opponent active being healthy.
    if me.active and me.active[0] is not None:
        val += me.active[0].hp * 1.0
    if op.active and op.active[0] is not None:
        val -= op.active[0].hp * 1.5  # pressure on their active is good
    # Card advantage (rough).
    val += me.handCount * 5.0
    return val


def _legal_fallback(select):
    """A structurally-legal selection: the first minCount distinct option indices."""
    n = len(select.option)
    k = max(1, select.minCount) if n else 0
    k = min(k, n)
    return list(range(k))


def search_plan(obs_dict, obs):
    """Optional single-turn forward search using the engine's forward model.

    Strategy (kept simple and robust): for the top few first-actions ranked by
    the heuristic, force each one, then GREEDILY complete the rest of our turn
    inside the simulator, and score the resulting board with evaluate_state().
    Pick the first-action whose rollout yields the best board. This turns the
    pure 1-ply greedy policy into "1-step lookahead + greedy completion".

    Returns a list[int] selection, or None to defer to the heuristic.

    NOTE: search_begin's exact input semantics depend on your SDK build. We pass
    obs.search_begin_input when present. Everything is wrapped so any failure
    cleanly defers to the heuristic -- the agent can never crash from search.
    """
    import time

    if not (_SEARCH_OK and USE_SEARCH):
        return None
    select = obs.select
    if select is None or select.context != SelectContext.MAIN:
        return None

    t0 = time.time()
    sbi = getattr(obs, "search_begin_input", None) or obs_dict.get("search_begin_input")
    if sbi is None:
        return None

    base_order = heuristic_agent(obs)  # heuristic ranking of first actions
    candidates = base_order[:SEARCH_MAX_CANDIDATES]

    best_idx, best_val = None, float("-inf")
    for first in candidates:
        if time.time() - t0 > SEARCH_TIME_BUDGET:
            break
        sid = None
        try:
            # Begin a fresh determinized search rooted at the current state.
            res = search_begin(sbi)  # SDK-specific; see local_harness smoke test
            if getattr(res, "error", 0) != 0 or res.state is None:
                return None
            sid = res.state.searchId
            cur = res.state.observation

            # Apply our chosen first action, then greedily finish the turn.
            sel = [first]
            steps = 0
            while steps < 40:
                ar = search_step(sid, sel)
                if getattr(ar, "error", 0) != 0 or ar.state is None:
                    break
                cur = ar.state.observation
                # Stop when the turn is no longer ours, or game ended.
                if cur.select is None or cur.current is None:
                    break
                if cur.current.result is not None and cur.current.result != -1:
                    break
                if cur.current.yourIndex != obs.current.yourIndex:
                    break
                if cur.select.context != SelectContext.MAIN:
                    # sub-selection: greedily resolve it
                    sub = heuristic_agent(cur)
                    sel = sub[: max(1, cur.select.minCount)]
                    steps += 1
                    continue
                # MAIN again -> greedily pick best; if best is END, finish.
                nxt = heuristic_agent(cur)
                sel = [nxt[0]]
                steps += 1
                if cur.select.option[nxt[0]].type == OptionType.END:
                    ar = search_step(sid, sel)
                    if ar.state is not None:
                        cur = ar.state.observation
                    break

            val = evaluate_state(cur)
            if val > best_val:
                best_val, best_idx = val, first
        except Exception:
            return None
        finally:
            try:
                if sid is not None:
                    search_release(sid)
            except Exception:
                pass

    if best_idx is None:
        return None
    # Put the chosen first action at the front; rest in heuristic order.
    rest = [i for i in base_order if i != best_idx]
    return [best_idx] + rest


# ============================================================================
# TOP-LEVEL AGENT  (crash-safe wrapper)
# ============================================================================
def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        # Cannot even parse -> if this is deck selection, return the deck.
        if obs_dict.get("select") is None:
            return my_deck
        return [0]

    # Initial deck selection.
    if obs.select is None:
        return my_deck

    select = obs.select
    try:
        # Optional lookahead (off by default; safe fallback inside).
        ordered = None
        if USE_SEARCH:
            ordered = search_plan(obs_dict, obs)
        if ordered is None:
            ordered = heuristic_agent(obs)

        # Respect minCount/maxCount and option bounds.
        n = len(select.option)
        ordered = [i for i in ordered if 0 <= i < n]
        if not ordered:
            return _legal_fallback(select)
        k = min(select.maxCount, n)
        k = max(k, min(max(1, select.minCount), n))
        return ordered[:k]
    except Exception:
        # Anything unexpected -> never crash, never forfeit on an exception.
        return _legal_fallback(select)
