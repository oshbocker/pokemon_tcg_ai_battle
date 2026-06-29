"""Vendored Kaggle pool agent: Mega Starmie ex + Cinderace (Water tempo).

A torch-free LEAGUE opponent for the asymmetric self-play pool — the #2 top-ladder
archetype (Mega Starmie ex), which was previously absent from `mixed_pool.json`.
Loaded by dist_worker as the `kaggle:starmie` opponent spec.

Provenance: the deck is our replay-extracted, engine-validated top-ladder Starmie
list (`agent/decks/mega_starmie.csv`, copied to the sibling `starmie_deck.csv`). The
two public Kaggle Starmie kernels — masamikobayashi *"Prize Card Tracking: 1300+
Starmie"* (gold-medal write-up; only its `PrizeTracker` helper is shared) and
map1e114514 *"Starmie Cinderace Budew Skill Agent"* (a `skills/` write-up, no code) —
publish **no full agent**, so the rule engine here is written from scratch on the
repo's own clean scoring scaffold (mirrors the vendored `archaludon.py` idiom, same
generic board helpers) and tuned to the engine's authoritative card/attack data. The
two write-ups inform the game plan (Cinderace Explosiveness → Turbo Flare energy
acceleration → evolve Staryu into Mega Starmie ex → Jetting Blow / Nebula Beam, with
Crushing Hammer + Boss disruption).

**It carries ITS OWN deck** (`starmie_deck.csv`) read RELATIVE TO THIS FILE — never
from the cwd `deck.csv` — so multiple borrowed agents (each a different archetype
deck) can coexist in one worker (workers chdir to agent/). Exposes the standard
`agent(obs)->list[int]` contract and `my_deck`, behind a crash-safe wrapper that
always returns a legal in-bounds selection.

Deck (60): Staryu 1030 x3, Mega Starmie ex 1031 x3, Cinderace 666 x4, Buddy-Buddy
Poffin 1086 x4, Mega Signal 1145 x4, Pokegear 3.0 1122 x4, Salvatore 1189 x4,
Harlequin 1223 x2, Hilda 1225 x2, Lillie's Determination 1227 x4, Wally's Compassion
1229 x4, Crushing Hammer 1120 x4, Night Stretcher 1097 x2, Ultra Ball 1121 x1, Boss's
Orders 1182 x1, Hero's Cape 1159 x1, Basic Water Energy 3 x9, Ignition Energy 17 x4.

Attacks (authoritative): Cinderace Turbo Flare (965, [C]=50, accelerate up to 3 Basic
Energy from deck to Bench); Staryu Water Gun (1486, [W]=20); Mega Starmie ex Jetting
Blow (1487, [W]=120 + 50 to a Benched Pokemon); Nebula Beam (1488, [C][C][C]=210,
damage not affected by Weakness/Resistance or by effects on the target). Water-type
attacks double vs Pokemon with Weakness value 3 (Water); Nebula Beam never doubles.
"""

import os
import sys

try:
    ROOT = __file__
except NameError:
    ROOT = None
CG_PATH = "/kaggle_simulations/agent"
for p in ([os.path.dirname(os.path.abspath(ROOT))] if ROOT else []) + [CG_PATH]:
    if p and p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from cg.api import (
    AreaType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

# ── Card IDs ──

WATER_ENERGY = 3
IGNITION_ENERGY = 17
CINDERACE = 666
STARYU = 1030
MEGA_STARMIE_EX = 1031

POFFIN = 1086        # Buddy-Buddy Poffin: bench up to 2 Basic Pokemon (HP <= 70)
NIGHT_STRETCHER = 1097
CRUSHING_HAMMER = 1120
ULTRA_BALL = 1121
POKEGEAR = 1122      # Pokegear 3.0: search a Supporter
MEGA_SIGNAL = 1145   # Item: evolution enabler into Mega Starmie ex
HERO_CAPE = 1159
BOSS = 1182
SALVATORE = 1189     # Supporter: evolution/search enabler
HARLEQUIN = 1223     # Supporter: draw/search
HILDA = 1225         # Supporter: deck search
LILLIE = 1227        # Lillie's Determination: draw
WALLY = 1229         # Wally's Compassion: evolution enabler

EVOLUTION_SUPPORTERS = {SALVATORE, WALLY}
DRAW_SEARCH_SUPPORTERS = {LILLIE, HILDA, HARLEQUIN}

# Attacks
TURBO_FLARE = 965
WATER_GUN = 1486
JETTING_BLOW = 1487
NEBULA_BEAM = 1488

WATER_WEAKNESS = 3   # Pokemon with this Weakness value take double from Water attacks
_WATER_ATTACKS = {WATER_GUN, JETTING_BLOW}
_ATTACK_BASE_DMG = {TURBO_FLARE: 50, WATER_GUN: 20, JETTING_BLOW: 120, NEBULA_BEAM: 210}

CARD_DB = {c.cardId: c for c in all_card_data()}

# A Cinderace in hand mid-game is dead (no Scorbunny line — it only enters play via
# Explosiveness during setup), so it is ideal Ultra Ball / Salvatore discard fodder.
ALWAYS_SAFE_DISCARD = {CINDERACE}

_SETUP_ACTIVE_PRIORITY = {
    CINDERACE: (100000, "Active: Cinderace Explosiveness (energy engine)"),
    STARYU: (20000, "Active fallback: Staryu"),
}

ITEMS = {POFFIN, NIGHT_STRETCHER, CRUSHING_HAMMER, ULTRA_BALL, POKEGEAR, MEGA_SIGNAL, HERO_CAPE}


# ── Board helpers (deck-agnostic) ──

def read_deck_csv():
    # Read THIS agent's own deck relative to this file (workers chdir to agent/, so a
    # bare "deck.csv" would wrongly load the trainee's deck). The sibling is the only
    # trusted source; the Kaggle runtime path is a pure fallback. A deck that isn't
    # exactly 60 cards is rejected so we never silently field the wrong deck.
    here = os.path.dirname(os.path.abspath(__file__))
    for fp in (
        os.path.join(here, "starmie_deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ):
        if os.path.exists(fp):
            with open(fp) as f:
                ids = [int(x) for x in f.read().split() if x.strip()]
            if len(ids) == 60:
                return ids
    raise FileNotFoundError("starmie_deck.csv (a legal 60-card sibling deck)")


my_deck = read_deck_csv()


def get_card(obs, area, index, player_index):
    if area is None or index is None:
        return None
    ps = obs.current.players[player_index]
    if area == AreaType.DECK and obs.select and obs.select.deck is not None:
        return obs.select.deck[index] if index < len(obs.select.deck) else None
    if area == AreaType.HAND and ps.hand is not None:
        return ps.hand[index] if index < len(ps.hand) else None
    if area == AreaType.DISCARD:
        return ps.discard[index] if index < len(ps.discard) else None
    if area == AreaType.ACTIVE:
        return ps.active[index] if index < len(ps.active) else None
    if area == AreaType.BENCH:
        return ps.bench[index] if index < len(ps.bench) else None
    if area == AreaType.PRIZE:
        return ps.prize[index] if index < len(ps.prize) else None
    if area == AreaType.STADIUM:
        return obs.current.stadium[index] if index < len(obs.current.stadium) else None
    if area == AreaType.LOOKING and obs.current.looking is not None:
        return obs.current.looking[index] if index < len(obs.current.looking) else None
    return None


def option_card(obs, opt):
    yi = obs.current.yourIndex
    pi = opt.playerIndex if opt.playerIndex is not None else yi
    if opt.type == OptionType.PLAY:
        return get_card(obs, AreaType.HAND, opt.index, pi)
    return get_card(obs, opt.area, opt.index, pi)


def option_target(obs, opt):
    if opt.inPlayArea is None or opt.inPlayIndex is None:
        return None
    return get_card(obs, opt.inPlayArea, opt.inPlayIndex, obs.current.yourIndex)


def my_state(obs):
    return obs.current.players[obs.current.yourIndex]


def opp_state(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def active_pokemon(obs):
    ps = my_state(obs)
    return ps.active[0] if ps.active else None


def opp_active_pokemon(obs):
    ps = opp_state(obs)
    return ps.active[0] if ps.active else None


def opp_bench_pokemon(obs):
    return [p for p in opp_state(obs).bench if p]


def all_my_pokemon(obs):
    ps = my_state(obs)
    return [p for p in (ps.active + ps.bench) if p]


def hand_ids(obs):
    hand = my_state(obs).hand
    return [c.id for c in hand if c] if hand else []


def discard_ids(obs):
    return [c.id for c in (my_state(obs).discard or []) if c]


def energy_count(pokemon):
    if pokemon is None:
        return 0
    if getattr(pokemon, "energyCards", None) is not None:
        return len(pokemon.energyCards)
    return len(getattr(pokemon, "energies", []) or [])


def water_energy_count(pokemon):
    """Count Basic Water Energy attached (what Jetting Blow's [W] cost needs)."""
    if pokemon is None:
        return 0
    cards = getattr(pokemon, "energyCards", None) or []
    return sum(1 for c in cards if c and c.id == WATER_ENERGY)


def retreat_cost(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    return getattr(data, "retreatCost", 0) if data else 0


def damage_on(pokemon):
    if pokemon is None:
        return 0
    return max(0, getattr(pokemon, "maxHp", pokemon.hp) - pokemon.hp)


def has_tool(pokemon):
    return bool(getattr(pokemon, "tools", []) or [])


def count_in_play(obs, card_id):
    return sum(1 for p in all_my_pokemon(obs) if p.id == card_id)


def has_in_play(obs, card_id):
    return any(p.id == card_id for p in all_my_pokemon(obs))


def prize_value(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


def is_water_weak(pokemon):
    if pokemon is None:
        return False
    data = CARD_DB.get(pokemon.id)
    w = getattr(data, "weakness", None) if data else None
    if w is None:
        return False
    return getattr(w, "value", w) == WATER_WEAKNESS


def attack_damage(obs, attack_id, target=None):
    """Effective damage of one of our attacks against `target` (defaults to opp Active)."""
    if target is None:
        target = opp_active_pokemon(obs)
    base = _ATTACK_BASE_DMG.get(attack_id, 0)
    # Nebula Beam ignores Weakness; Water attacks double vs Water-weak Pokemon.
    if attack_id in _WATER_ATTACKS and is_water_weak(target):
        base *= 2
    return base


# ── Line-state helpers ──

def starmie_attackers(obs):
    """Mega Starmie ex in play that already meet an attack threshold (>=1 energy)."""
    return [p for p in all_my_pokemon(obs) if p.id == MEGA_STARMIE_EX and energy_count(p) >= 1]


def need_staryu(obs):
    """Fewer than 2 of the Staryu line (Staryu + Mega Starmie ex) in play."""
    return sum(1 for p in all_my_pokemon(obs) if p.id in {STARYU, MEGA_STARMIE_EX}) < 2


def need_starmie(obs):
    """Have a Staryu to evolve but fewer than 2 Mega Starmie ex attackers."""
    has_staryu = has_in_play(obs, STARYU)
    ex_count = count_in_play(obs, MEGA_STARMIE_EX)
    return has_staryu and ex_count < 2


def can_evolve_staryu_now(obs):
    """A Staryu that has been in play since before this turn can evolve this turn."""
    for p in all_my_pokemon(obs):
        if p.id == STARYU and not getattr(p, "appearThisTurn", True):
            return True
    return False


def best_attacker_route(obs):
    """The Mega Starmie ex we would most like to attack with, and whether it must
    retreat first. Prefer the Active if it can attack; else a benched one if the
    Active can pay retreat."""
    active = active_pokemon(obs)
    if active and active.id == MEGA_STARMIE_EX and energy_count(active) >= 1:
        return {"attacker": active, "needs_retreat": False}
    if active is None or getattr(obs.current, "retreated", False):
        return None
    if energy_count(active) < retreat_cost(active):
        return None
    for p in [b for b in my_state(obs).bench if b]:
        if p.id == MEGA_STARMIE_EX and energy_count(p) >= 3:
            return {"attacker": p, "needs_retreat": True}
    return None


def planned_attacks(obs):
    """Damage values an attack-ready Mega Starmie ex could deal this turn."""
    route = best_attacker_route(obs)
    if route is None:
        return []
    attacker = route["attacker"]
    e = energy_count(attacker)
    out = []
    if e >= 1:
        out.append(JETTING_BLOW)
    if e >= 3:
        out.append(NEBULA_BEAM)
    return out


def can_ko_with(obs, target):
    return any(attack_damage(obs, aid, target) >= target.hp for aid in planned_attacks(obs))


# ── Scoring ──

def score_setup(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ctx = obs.select.context

    if ctx == SelectContext.MULLIGAN:
        return (10000, "no mulligan") if opt.type == OptionType.NO else (0, "mulligan")
    if ctx == SelectContext.IS_FIRST:
        # Mega evolution + Cinderace acceleration favour going second.
        return (10000, "choose second") if opt.type == OptionType.NO else (0, "go first")
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        return _SETUP_ACTIVE_PRIORITY.get(cid, (1000, "unknown Active"))
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        # Unlike a one-tank deck, Starmie wants Staryu benched to evolve next turn.
        if cid == STARYU:
            return 15000, "bench Staryu (evolve target)"
        return 1000, "bench basic"
    return 0, "non-setup"


def score_play(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ids = hand_ids(obs)

    # ── Pokemon ──
    if cid == STARYU:
        return (18000 if need_staryu(obs) else 6000), "bench Staryu"
    if cid == MEGA_STARMIE_EX:
        # Played as a hand-evolution onto Staryu; route through EVOLVE scoring instead.
        return (17000 if need_starmie(obs) else 2000), "evolve to Mega Starmie ex"

    # ── Evolution enablers ──
    if cid == MEGA_SIGNAL:
        if need_starmie(obs) and can_evolve_staryu_now(obs):
            return 26000, "Mega Signal -> Mega Starmie ex"
        return -500, "save Mega Signal: no Staryu ready"

    # ── Items ──
    if cid in ITEMS:
        if cid == POFFIN:
            return (22000 if need_staryu(obs) else -300), "Buddy-Buddy Poffin (bench Staryu)"
        if cid == ULTRA_BALL:
            bench_empty = len([p for p in my_state(obs).bench if p]) == 0
            if need_staryu(obs) or bench_empty:
                return 20000, "Ultra Ball: search Pokemon"
            return -1000, "save Ultra Ball"
        if cid == POKEGEAR:
            if obs.current.supporterPlayed:
                return -300, "Pokegear: search a Supporter"
            return 15000, "Pokegear: dig for Supporter"
        if cid == NIGHT_STRETCHER:
            disc = discard_ids(obs)
            recover_target = (
                (MEGA_STARMIE_EX in disc and need_starmie(obs))
                or (STARYU in disc and need_staryu(obs))
                or (WATER_ENERGY in disc and not obs.current.energyAttached
                    and not starmie_attackers(obs))
            )
            return (16000 if recover_target else -500), "Night Stretcher"
        if cid == CRUSHING_HAMMER:
            opp = opp_active_pokemon(obs)
            return (8000 if opp and energy_count(opp) >= 1 else -800), "Crushing Hammer (strip energy)"
        if cid == HERO_CAPE:
            target = any(p.id == MEGA_STARMIE_EX and not has_tool(p) for p in all_my_pokemon(obs))
            return (9000 if target else -500), "Hero's Cape on Mega Starmie ex"
        return 12000, "play item"

    # ── Supporters (one per turn) ──
    if cid in EVOLUTION_SUPPORTERS:
        if obs.current.supporterPlayed:
            return -1000, "Supporter already used"
        if need_starmie(obs) and can_evolve_staryu_now(obs):
            return 24000, "evolution Supporter -> Mega Starmie ex"
        if need_staryu(obs):
            return 9000, "evolution Supporter: dig"
        return 4000, "evolution Supporter"

    if cid == BOSS:
        if obs.current.supporterPlayed:
            return -1000, "Supporter already used"
        attacks = planned_attacks(obs)
        if not attacks:
            return -500, "save Boss: no attacker ready"
        opp_act = opp_active_pokemon(obs)
        remaining = len(my_state(obs).prize)
        # Lethal: drag a benched Pokemon we can KO that takes the last prize(s).
        for target in opp_bench_pokemon(obs):
            if can_ko_with(obs, target) and prize_value(target) >= remaining:
                return 20000, "LETHAL Boss"
        # Otherwise pull the most valuable benched target we can KO.
        best, reason = -500, "save Boss"
        for target in opp_bench_pokemon(obs):
            if can_ko_with(obs, target):
                s = 5000 + prize_value(target) * 800 + energy_count(target) * 100
                if s > best:
                    best, reason = s, "Boss: pull KO target"
        if best <= 0 and opp_act and not can_ko_with(obs, opp_act):
            # Active is a wall we can't KO — drag up something softer to keep tempo.
            for target in opp_bench_pokemon(obs):
                if target.hp < opp_act.hp:
                    return 4000, "Boss: drag softer target"
        return best, reason

    if cid in DRAW_SEARCH_SUPPORTERS:
        if obs.current.supporterPlayed:
            return -1000, "Supporter already used"
        # Hold the turn's Supporter for evolution if a Staryu is ready and an enabler
        # is also in hand; otherwise dig.
        if can_evolve_staryu_now(obs) and need_starmie(obs) and (EVOLUTION_SUPPORTERS & set(ids)):
            return 3000, "draw Supporter: defer to evolution"
        return 6500, "draw/search Supporter"

    return 1000, "generic play"


def score_evolve(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    tid = target.id if target else None
    if cid == MEGA_STARMIE_EX and tid == STARYU:
        target_is_active = opt.inPlayArea == AreaType.ACTIVE
        if count_in_play(obs, MEGA_STARMIE_EX) == 0:
            return (28000 if target_is_active else 26000), "evolve into first Mega Starmie ex"
        # Prefer evolving an energized Staryu (keeps the attack online sooner).
        return 14000 + energy_count(target) * 1500, "evolve into backup Mega Starmie ex"
    return 10000, "generic evolution"


def attach_target_score(obs, energy_id, target, area):
    """Value of attaching `energy_id` to `target`. Water -> Mega Starmie ex (enables
    Jetting Blow at 1, Nebula Beam at 3); Cinderace wants exactly 1 for Turbo Flare."""
    if target is None:
        return -1000
    cid = target.id
    e = energy_count(target)
    is_water = energy_id == WATER_ENERGY

    if cid == MEGA_STARMIE_EX:
        if e >= 3:
            return -3000  # already at Nebula Beam cost
        score = 9000
        score += {0: 6000, 1: 4000, 2: 5000}.get(e, 0)  # 1->2->3 keeps pushing to 210
        if is_water and e == 0:
            score += 4000  # first Water unlocks Jetting Blow immediately
        score += 1500 if area == AreaType.ACTIVE else 0
        return score
    if cid == STARYU:
        if e >= 1:
            return -2000
        # A single Water on Staryu carries over to Jetting Blow after evolving.
        return 4000 if is_water else 1500
    if cid == CINDERACE:
        if e >= 1:
            return -3000  # 1 energy already powers Turbo Flare
        # Prefer Ignition on Cinderace; save Basic Water for the Starmie attacker.
        return (5000 if not is_water else 2500) + (1500 if area == AreaType.ACTIVE else 0)
    return 500


def score_attach(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    tid = target.id if target else None

    if cid == HERO_CAPE:
        if tid == MEGA_STARMIE_EX and target and not has_tool(target):
            return 11000, "Hero's Cape on Mega Starmie ex"
        return -1000, "save Hero's Cape"

    if cid not in {WATER_ENERGY, IGNITION_ENERGY}:
        return -500, "skip non-energy attach"
    if obs.current.energyAttached:
        return -1000, "already attached this turn"
    return attach_target_score(obs, cid, target, opt.inPlayArea), "attach energy"


def score_retreat(obs, opt):
    route = best_attacker_route(obs)
    if route and route["needs_retreat"]:
        return 13000, "retreat to attack-ready Mega Starmie ex"
    active = active_pokemon(obs)
    # Retreat a stranded Cinderace (its job is done once Turbo Flare has fired and a
    # Starmie attacker is online) only if it would not waste a needed attack.
    if active and active.id == CINDERACE and starmie_attackers(obs) and not obs.current.retreated:
        return 6000, "retreat spent Cinderace"
    return -100, "avoid retreat"


def attack_score(obs, opt):
    aid = getattr(opt, "attackId", None)
    opp = opp_active_pokemon(obs)
    dmg = attack_damage(obs, aid, opp)
    score = dmg
    if opp and dmg >= opp.hp:
        score += 5000 + prize_value(opp) * 500  # secure the KO
    if aid == NEBULA_BEAM:
        score += 200  # ignores damage-reduction / wall effects
    if aid == JETTING_BLOW and opp_bench_pokemon(obs):
        score += 60  # +50 bench snipe is incidental upside
    if aid == TURBO_FLARE:
        # The setup engine: worth more than ending the turn while a benched Starmie
        # still needs energy, but never over a real KO swing.
        if any(p.id in {STARYU, MEGA_STARMIE_EX} and energy_count(p) < 3
               for p in my_state(obs).bench if p):
            score += 300
    return score, "attack"


def score_to_hand(obs, opt):
    """Search/draw selection (TO_HAND): pick the cards that advance the Starmie line."""
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)

    if cid == STARYU and need_staryu(obs):
        return 22000, "take Staryu"
    if cid == MEGA_STARMIE_EX and need_starmie(obs) and MEGA_STARMIE_EX not in ids:
        return 21000, "take Mega Starmie ex"
    if cid == MEGA_SIGNAL and MEGA_SIGNAL not in ids and need_starmie(obs):
        return 18000, "take Mega Signal"
    if cid in EVOLUTION_SUPPORTERS and not obs.current.supporterPlayed:
        return 14000, "take evolution Supporter"
    if cid == POFFIN and need_staryu(obs):
        return 13000, "take Poffin"
    if cid == WATER_ENERGY:
        return 9000, "take Water Energy"
    if cid == IGNITION_ENERGY:
        return 7000, "take Ignition Energy"
    if cid in DRAW_SEARCH_SUPPORTERS and not obs.current.supporterPlayed:
        return 6500, "take Supporter"
    if cid == BOSS:
        return 5000, "take Boss"
    if cid == ULTRA_BALL:
        return 4000, "take Ultra Ball"
    if cid == CINDERACE:
        return -2000, "skip dead Cinderace"
    return 1000, "generic take"


def score_discard(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)

    if cid in ALWAYS_SAFE_DISCARD:
        return 12000, "discard dead Cinderace"
    if cid == IGNITION_ENERGY and ids.count(IGNITION_ENERGY) > 1:
        return 9000, "discard surplus Ignition"
    if cid == WATER_ENERGY and ids.count(WATER_ENERGY) > 2:
        return 8000, "discard surplus Water"
    if cid in {POKEGEAR, CRUSHING_HAMMER} and ids.count(cid) > 1:
        return 8500, "discard duplicate item"
    if cid in DRAW_SEARCH_SUPPORTERS and ids.count(cid) > 1:
        return 8000, "discard duplicate Supporter"
    if cid == MEGA_STARMIE_EX:
        return -5000, "keep Mega Starmie ex"
    if cid == STARYU:
        return -4000, "keep Staryu"
    if cid == MEGA_SIGNAL and need_starmie(obs):
        return -3000, "keep Mega Signal"
    if cid == WATER_ENERGY:
        return -1000, "keep last Water"
    return 1000, "generic discard"


def score_target(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ctx = obs.select.context

    if ctx == SelectContext.ATTACH_TO:
        if cid in {WATER_ENERGY, IGNITION_ENERGY}:
            return 5000, "energy target"
        return 1000, "attach target"

    if ctx == SelectContext.ATTACH_FROM:
        # Accelerating Basic Energy onto the board (e.g. Turbo Flare): load the
        # benched Starmie line toward an attack.
        if card and cid == MEGA_STARMIE_EX and energy_count(card) < 3:
            return 9000 + energy_count(card) * 500, "accelerate onto Mega Starmie ex"
        if card and cid == STARYU and energy_count(card) == 0:
            return 6000, "accelerate onto Staryu"
        if card and cid == CINDERACE:
            return -2000, "skip Cinderace for acceleration"
        return attach_target_score(obs, WATER_ENERGY, card, getattr(opt, "area", None)), "effect attach"

    if ctx in {SelectContext.TO_FIELD, SelectContext.TO_BENCH}:
        if cid == MEGA_STARMIE_EX:
            return 18000, "target Mega Starmie ex"
        if cid == STARYU:
            return 16000, "target Staryu"
        if cid == CINDERACE:
            return 2000, "avoid Cinderace"

    if ctx == SelectContext.HEAL:
        return (20000 + damage_on(card), "heal Mega Starmie ex") if cid == MEGA_STARMIE_EX \
            else (damage_on(card), "heal")

    if ctx in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
        yi = obs.current.yourIndex
        pi = getattr(opt, "playerIndex", yi)
        if pi != yi and card:
            # Opponent-side gust target (Boss): prefer a Pokemon we can KO.
            pv = prize_value(card)
            te = energy_count(card)
            if can_ko_with(obs, card):
                return 20000 + pv * 3000 + te * 100, "Boss: KO target"
            return 5000 + pv * 1000 + te * 200, "Boss: drag target"
        # Our side: promote an attack-ready Mega Starmie ex, else Cinderace (retreat 0).
        if cid == MEGA_STARMIE_EX:
            return 16000 + energy_count(card) * 500, "promote Mega Starmie ex"
        if cid == CINDERACE:
            return 12000, "promote Cinderace (retreat 0)"
        if cid == STARYU:
            return 8000, "promote Staryu"
        return 1000, "generic promote"

    if ctx == SelectContext.DAMAGE:
        hp = getattr(card, "hp", 999) if card else 999
        return 10000 - hp, "damage: lowest HP"

    return 1000, "generic target"


_MAIN_DISPATCH = {
    OptionType.PLAY: score_play,
    OptionType.EVOLVE: score_evolve,
    OptionType.ATTACH: score_attach,
    OptionType.RETREAT: score_retreat,
}


def score_option(obs, opt):
    ctx = obs.select.context

    if ctx in {SelectContext.IS_FIRST, SelectContext.MULLIGAN,
               SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON}:
        return score_setup(obs, opt)

    if opt.type in {OptionType.YES, OptionType.NO}:
        if ctx == SelectContext.IS_FIRST:
            return score_setup(obs, opt)
        if ctx == SelectContext.ACTIVATE:
            # Cinderace Explosiveness (and other beneficial activations): always accept.
            return (100000, "activate") if opt.type == OptionType.YES else (-100000, "never decline")
        return (1, "yes") if opt.type == OptionType.YES else (0, "no")

    if opt.type == OptionType.NUMBER:
        return (opt.number or 0), "number"

    if ctx == SelectContext.MAIN:
        fn = _MAIN_DISPATCH.get(opt.type)
        if fn:
            return fn(obs, opt)
        if opt.type == OptionType.ABILITY:
            return 1, "ability"
        if opt.type == OptionType.ATTACK:
            return attack_score(obs, opt)
        if opt.type == OptionType.END:
            return 0, "end turn"
        return 500, "generic MAIN"

    if ctx == SelectContext.TO_HAND:
        return score_to_hand(obs, opt)
    if ctx in {SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD}:
        return score_discard(obs, opt)
    if ctx in {SelectContext.ATTACH_TO, SelectContext.TO_FIELD, SelectContext.TO_BENCH,
               SelectContext.ATTACH_FROM, SelectContext.SWITCH, SelectContext.TO_ACTIVE,
               SelectContext.HEAL, SelectContext.DAMAGE}:
        return score_target(obs, opt)
    if ctx == SelectContext.ATTACK:
        return attack_score(obs, opt)
    if opt.type == OptionType.CARD:
        return score_to_hand(obs, opt)
    if opt.type == OptionType.ENERGY:
        return 1000, "energy"
    if opt.type == OptionType.END:
        return 0, "end"
    return 100, "fallback"


# ── Choose & Agent ──

def choose_options(obs):
    scored = []
    for i, opt in enumerate(obs.select.option):
        try:
            score, reason = score_option(obs, opt)
        except Exception as e:
            score, reason = -999999, f"error {type(e).__name__}: {e}"
        scored.append((score, i, reason))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    selected = []
    for score, i, reason in scored:
        if len(selected) >= obs.select.maxCount:
            break
        if score < 0 and len(selected) >= obs.select.minCount:
            continue
        selected.append(i)

    if len(selected) < obs.select.minCount:
        selected = [i for _, i, _ in scored[:obs.select.minCount]]

    return selected


def agent(obs_dict):
    # Never crash / forfeit: parse + deck-selection are guarded, and the deck is the
    # cached `my_deck` (read once at import), not a per-call file read.
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return my_deck if obs_dict.get("select") is None else [0]
    if obs.select is None:
        return my_deck
    if not obs.select.option:
        return []
    try:
        return choose_options(obs)
    except Exception:
        n = len(obs.select.option)
        k = min(max(obs.select.minCount, 1), n)
        return list(range(k))
