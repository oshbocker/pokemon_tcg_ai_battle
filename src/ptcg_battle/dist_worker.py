"""Torch-free env-worker for the distributed collector (Phase 3, P3.1).

Deliberately split out of `dist_collector.py` so the spawned worker processes
import *only this* (engine + the numpy-only encoder) and never pull in torch — the
central process is the sole torch/GPU owner. That keeps each worker's RAM and
spawn cost low (the L4 box is vCPU- and RAM-constrained), and enforces the design
rule that workers are pure engine steppers.

A worker owns one `battle_ptr`, plays one game at a time, and for every decision
that belongs to the model it ships an `EncodedObs` to the central inference loop
and blocks on the chosen action. Fixed opponents (`random`/`first`/`heuristic`)
are stepped locally. See `dist_collector.DistributedCollector` for the protocol.
"""

from __future__ import annotations

import contextlib
import os
import queue as queue_mod
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO / "agent"

# Request-queue message tags (worker -> central).
DECIDE = "decide"
DONE = "done"
# Task-queue tags (central -> worker).
PLAY = "play"
STOP = "stop"


def _make_worker_opponent(spec: str, deck: list[int], rng: random.Random, to_oc):
    """A torch-free fixed opponent stepped inside the worker (mirrors ppo._make_fixed_opponent)."""
    if spec == "heuristic":
        import importlib.util

        s = importlib.util.spec_from_file_location("opp_heur", AGENT_DIR / "main.py")
        assert s is not None and s.loader is not None
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod.agent

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


def worker_main(wid, opponent, deck, max_steps, task_q, req_q, resp_q):  # pragma: no cover
    """Run forever: pull a game token, play it (asking central for model moves), repeat.

    Runs in a spawned subprocess, so it imports the engine + encoder itself and is
    never measured by the parent's coverage.
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
    opp_fn = None if opponent == "self" else _make_worker_opponent(opponent, deck, rng, to_oc)

    while True:
        tok = task_q.get()
        if tok[0] == STOP:
            return
        _, _gidx, model_seat, opp_seed = tok
        if opp_fn is not None:
            rng.seed(opp_seed)
        _drain_stale(resp_q)
        result = None
        obs = None
        try:
            obs, _ = battle_start(deck, deck)
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
                    if opp_fn is not None and seat != model_seat:
                        obs = battle_select(opp_fn(obs))
                        continue
                    enc = encode_observation(obs)
                    req_q.put((DECIDE, wid, seat, enc))
                    action = resp_q.get()
                    obs = battle_select(action)
        except Exception:  # noqa: BLE001 — a crash forfeits this game, pool keeps going
            result = None
        finally:
            if obs is not None:
                with contextlib.suppress(Exception):
                    battle_finish()
        req_q.put((DONE, wid, result))
