"""Shared fixtures for the test suite.

The cabt engine is a global singleton loaded via ctypes and resolves `libcg.so`
+ `deck.csv` relative to the agent directory, so engine-touching fixtures chdir
into `agent/` for the duration of collection and restore cwd afterwards. Raw obs
dicts are deep-detached from the engine before being handed to tests, so the
tests themselves never touch the singleton (and stay order-independent).
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO / "agent"


def _collect_observations(n_games: int = 8, max_steps: int = 2000) -> list[dict]:
    """Play `n_games` heuristic mirror games, returning one detached raw obs dict
    per distinct (context, maxCount>1) bucket seen — broad SelectContext coverage
    plus multi-pick examples and the largest option list encountered."""
    prev_cwd = Path.cwd()
    os.chdir(AGENT_DIR)
    sys.path.insert(0, str(AGENT_DIR))
    try:
        from cg.api import to_observation_class  # type: ignore[reportMissingImports]
        from cg.game import (  # type: ignore[reportMissingImports]
            battle_finish,
            battle_select,
            battle_start,
        )

        spec = importlib.util.spec_from_file_location("our_agent", AGENT_DIR / "main.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        deck = mod.my_deck

        by_key: dict[tuple, dict] = {}
        for _ in range(n_games):
            obs, _start = battle_start(deck, deck)
            if obs is None:
                continue
            for _ in range(max_steps):
                oc = to_observation_class(obs)
                res = oc.current.result if oc.current is not None else -1
                if res is not None and res >= 0:
                    break
                sel = obs["select"]
                key = (sel["context"], sel["maxCount"] > 1)
                by_key.setdefault(key, copy.deepcopy(obs))
                obs = battle_select(mod.agent(obs))
            battle_finish()
        return list(by_key.values())
    finally:
        os.chdir(prev_cwd)


@pytest.fixture(scope="session")
def obs_samples() -> list[dict]:
    samples = _collect_observations()
    assert len(samples) >= 8, f"expected broad context coverage, got {len(samples)}"
    return samples


@pytest.fixture(scope="session")
def engine_card_attack() -> tuple[list[int], list[int]]:
    """(card_ids, attack_ids) from the live engine, for vocab-bound assertions."""
    prev_cwd = Path.cwd()
    os.chdir(AGENT_DIR)
    sys.path.insert(0, str(AGENT_DIR))
    try:
        from cg.api import all_attack, all_card_data  # type: ignore[reportMissingImports]

        return ([c.cardId for c in all_card_data()], [a.attackId for a in all_attack()])
    finally:
        os.chdir(prev_cwd)
