"""Local self-play validation harness.

Mirrors the Kaggle validation episode (agent vs. a copy of itself) and a quick
vs-random sanity check, running directly on the cabt engine via cg.game.
Run from the repo root:

    uv run python scripts/local_selfplay.py
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))  # so `import cg` resolves to agent/cg
os.chdir(AGENT_DIR)  # so the agent finds deck.csv (relative path) the same way it does on Kaggle

from cg.api import to_observation_class  # type: ignore[reportMissingImports]  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # type: ignore[reportMissingImports]  # noqa: E402


def load_agent():
    spec = importlib.util.spec_from_file_location("our_agent", AGENT_DIR / "main.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legal_k(oc):
    n = len(oc.select.option)
    k = min(oc.select.maxCount, n)
    if k < oc.select.minCount:
        k = min(oc.select.minCount, n)
    return max(1, k), n


def make_random_agent(deck):
    def random_agent(obs):
        oc = to_observation_class(obs)
        if oc.select is None:
            return deck
        k, n = legal_k(oc)
        return random.sample(range(n), min(k, n))

    return random_agent


def play_game(a0, d0, a1, d1, max_steps=4000):
    obs = None
    try:
        obs, start = battle_start(d0, d1)
        if obs is None:
            return None, f"start_failed:{getattr(start, 'errorType', '?')}"
        for _ in range(max_steps):
            oc = to_observation_class(obs)
            res = oc.current.result if oc.current is not None else -1
            if res is not None and res >= 0:
                return res, ""
            active = a0 if oc.current.yourIndex == 0 else a1
            obs = battle_select(active(obs))
        return None, "max_steps"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    finally:
        if obs is not None:
            try:
                battle_finish()
            except Exception:  # noqa: BLE001
                pass


def series(label, a0, d0, a1, d1, n):
    s0 = s1 = draws = errs = 0
    first_err = ""
    for _ in range(n):
        res, err = play_game(a0, d0, a1, d1)
        if err:
            errs += 1
            first_err = first_err or err
        elif res == 0:
            s0 += 1
        elif res == 1:
            s1 += 1
        else:
            draws += 1
    print(
        f"{label:26s} n={n}  seat0={s0}  seat1={s1}  draws={draws}  errors={errs}"
        + (f"  [{first_err}]" if first_err else "")
    )
    return errs


def main():
    our = load_agent()
    deck = our.my_deck
    assert our.agent({"select": None}) == deck, "agent must return the 60-card deck on init"
    print(f"OK  agent imports; deck has {len(deck)} cards, {len(set(deck))} unique.\n")

    rnd = make_random_agent(deck)
    t0 = time.time()
    total_err = 0
    total_err += series("mirror (ours vs ours)", our.agent, deck, our.agent, deck, 20)
    total_err += series("ours (seat0) vs random", our.agent, deck, rnd, deck, 10)
    total_err += series("random (seat0) vs ours", rnd, deck, our.agent, deck, 10)
    print(f"\nelapsed: {time.time() - t0:.1f}s")
    print(
        "PASS: validation-safe (0 errors)."
        if total_err == 0
        else f"FAIL: {total_err} errors — fix before submitting."
    )
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
