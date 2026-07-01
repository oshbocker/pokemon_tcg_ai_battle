"""Torch-free env-worker for the distributed collector (Phase 3, P3.1 / P3.4 league).

Deliberately split out of `dist_collector.py` so the spawned worker processes
import *only this* (engine + the numpy-only encoder) and never pull in torch — the
central process is the sole torch/GPU owner. That keeps each worker's RAM and
spawn cost low (the L4 box is vCPU- and RAM-constrained), and enforces the design
rule that workers are pure engine steppers.

A worker owns one `battle_ptr`, plays one game at a time, and is told per game (via
the PLAY token) what the *opponent* seat is — `opp_spec`:

  * ``"self"``            — opponent seat is the **current** policy (both seats
                           trained); routed to central with policy id ``"cur"``.
  * ``"model:<id>"``      — opponent seat is a **frozen past checkpoint** in the
                           central pool; routed to central with that policy id,
                           **not** trained.
  * a fixed-agent spec    — ``"random"`` / ``"first"`` / ``"heuristic"`` /
                           ``"kaggle:<name>"`` — stepped **locally** in the worker
                           (torch-free), not trained.

The acting (model-seat) decisions are always the current policy (``"cur"``). This
per-game opponent mix is the league (P3.4): it breaks the pure-self-play
mutual-determinism collapse by facing the trainee with diverse, competent,
non-degenerate opponents. See `dist_collector.DistributedCollector`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import queue as queue_mod
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO / "agent"
KAGGLE_AGENT_DIR = AGENT_DIR / "kaggle_agents"

# Request-queue message tags (worker -> central).
DECIDE = "decide"
DONE = "done"
# Task-queue tags (central -> worker).
PLAY = "play"
STOP = "stop"
# Policy id for the current (trained) net in a central decide request.
CUR = "cur"


def _load_agent_module(path: Path, name: str):
    """Import a standalone agent module (its own globals) for `agent(obs)->action`."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_fixed_agent(spec: str, deck: list[int], rng: random.Random, to_oc):
    """Resolve a fixed-agent `opp_spec` to a torch-free `agent(obs)->list[int]`.

    `random`/`first` are stepped here; `heuristic` loads `agent/main.py`; a
    `kaggle:<name>` spec loads `agent/kaggle_agents/<name>.py` — both expose the
    standard `agent(obs)` contract and keep their own module-global turn state."""
    if spec == "heuristic":
        return _load_agent_module(AGENT_DIR / "main.py", "opp_heuristic").agent
    if spec.startswith("kaggle:"):
        name = spec[len("kaggle:") :]
        return _load_agent_module(KAGGLE_AGENT_DIR / f"{name}.py", f"opp_{name}").agent

    def fn(obs_dict):
        oc = to_oc(obs_dict)
        if oc.select is None:
            return deck
        n = len(oc.select.option)
        k = max(1, min(oc.select.maxCount, n))
        if k < oc.select.minCount:
            k = min(oc.select.minCount, n)
        return rng.sample(range(n), min(k, n)) if spec == "random" else list(range(min(k, n)))

    return fn


def _drain_stale(resp_q) -> None:
    """Clear any response left over from a game that died mid-decision."""
    while True:
        try:
            resp_q.get_nowait()
        except queue_mod.Empty:
            return


def worker_main(wid, fixed_specs, deck_self, max_steps, task_q, req_q, resp_q):  # pragma: no cover
    """Run forever: pull a game token, play it (per-game opponent), repeat.

    `deck_self` is the **trainee** deck (the deck the current/past policies pilot);
    it always seats the model. The opponent's deck arrives per game in the PLAY
    token (`opp_deck`): for ``self``/``model:`` opponents it is ``None`` (they pilot
    `deck_self` too — true self/past mirror); for a fixed/Kaggle opponent it is THAT
    agent's own deck, so the match is asymmetric (`battle_start(deck_self, opp_deck)`).

    `fixed_specs` is the set of locally-stepped opponent specs this worker may be
    asked to play (built once at startup); `"self"` / `"model:<id>"` opponents are
    routed to the central GPU loop instead. Runs in a spawned subprocess, so it
    imports the engine + encoder itself and is never measured by parent coverage.
    """
    os.chdir(AGENT_DIR)
    sys.path.insert(0, str(AGENT_DIR))
    from cg.api import to_observation_class as to_oc  # type: ignore[reportMissingImports]
    from cg.game import (  # type: ignore[reportMissingImports]
        battle_finish,
        battle_select,
        battle_start,
    )

    from .encoding import encode_observation

    rng = random.Random(7919 + wid)
    fixed_agents = {s: _make_fixed_agent(s, deck_self, rng, to_oc) for s in fixed_specs}

    while True:
        tok = task_q.get()
        if tok[0] == STOP:
            return
        _, _gidx, model_seat, opp_seed, opp_spec, opp_deck = tok
        rng.seed(opp_seed)
        opp_fixed = fixed_agents.get(opp_spec)  # None unless a locally-stepped opponent
        # opp seat policy id when it is routed to central (self → current net):
        opp_policy = CUR if opp_spec == "self" else opp_spec  # else "model:<id>"
        # Seat the decks: trainee at model_seat, opponent (its own deck, or the
        # trainee deck for self/past mirrors) at the other seat. The deck can't react
        # to play order, so the model_seat side-swap is what cancels first-player bias.
        seat_decks = [deck_self, deck_self]
        seat_decks[1 - model_seat] = opp_deck if opp_deck is not None else deck_self
        _drain_stale(resp_q)
        result = None
        obs = None
        try:
            obs, _ = battle_start(seat_decks[0], seat_decks[1])
            if obs is not None:
                for _ in range(max_steps):
                    oc = to_oc(obs)
                    cur = oc.current
                    if cur is None:
                        break
                    if cur.result is not None and cur.result >= 0:
                        result = cur.result
                        break
                    seat = cur.yourIndex
                    if seat != model_seat and opp_fixed is not None:  # local fixed opponent
                        obs = battle_select(opp_fixed(obs))
                        continue
                    policy = CUR if seat == model_seat else opp_policy
                    enc = encode_observation(obs)
                    req_q.put((DECIDE, wid, seat, policy, enc))
                    action = resp_q.get()
                    obs = battle_select(action)
        except Exception:  # noqa: BLE001 — a crash forfeits this game, pool keeps going
            result = None
        finally:
            if obs is not None:
                with contextlib.suppress(Exception):
                    battle_finish()
        # Report the opponent + the learner's seat so the collector can tally per-
        # opponent outcomes (realized training share, win-rate, and forfeits).
        req_q.put((DONE, wid, result, opp_spec, model_seat))
