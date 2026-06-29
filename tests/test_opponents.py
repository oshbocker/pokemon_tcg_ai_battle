"""Opponent-manifest (league-as-data) tests — torch-free.

The manifest is the pivot-ready knob: opponents + their own decks chosen by config,
never code. These checks pin the resolution rules (kaggle:<name> → sibling deck,
heuristic → agent/deck.csv, random → mirror/None) and the default pool's integrity.
"""

from __future__ import annotations

from ptcg_battle.opponents import (
    default_deck_path,
    load_manifest,
    manifest_to_league_args,
    read_deck,
    resolve_deck,
)


def test_default_deck_path_rules():
    archa = default_deck_path("kaggle:archaludon")
    heur = default_deck_path("heuristic")
    assert archa is not None and archa.name == "archaludon_deck.csv"
    assert heur is not None and heur.name == "deck.csv"
    assert default_deck_path("random") is None
    assert default_deck_path("first") is None


def test_resolve_deck_kaggle_uses_sibling():
    deck = resolve_deck("kaggle:dragapult", None)
    assert deck is not None and len(deck) == 60
    # explicit override wins over the default sibling
    override = resolve_deck("kaggle:dragapult", "agent/decks/iono.csv")
    assert override == read_deck("agent/decks/iono.csv")


def test_resolve_deck_random_is_mirror():
    assert resolve_deck("random", None) is None  # mirror the trainee deck


def test_default_pool_manifest_loads_and_resolves():
    opps = load_manifest("agent/opponents/mixed_pool.json")
    specs = {o.spec for o in opps}
    assert {"kaggle:archaludon", "kaggle:dragapult"} <= specs
    # every kaggle opponent resolves a real 60-card own-deck
    for o in opps:
        if o.spec.startswith("kaggle:"):
            assert o.deck is not None and len(o.deck) == 60
    extra, decks = manifest_to_league_args(opps)
    assert all(w > 0 for _, w in extra)
    assert "kaggle:archaludon" in decks and len(decks["kaggle:archaludon"]) == 60
    # self/model are trainer-managed, never in the manifest
    assert "self" not in specs and not any(s.startswith("model:") for s in specs)
