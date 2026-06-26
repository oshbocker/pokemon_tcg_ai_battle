"""Phase 0 throughput benchmark for the cabt engine (self-play RL feasibility).

Measures the metric that gates the whole RL plan: how many **agent decisions per
second** we can generate via process-level self-play. The cabt engine is a global
singleton (`cg.game.Battle.battle_ptr`), so parallelism is across *processes*,
never threads — this script sweeps a worker-count grid and reports aggregate
games/sec and decisions/sec.

It deliberately uses a near-zero-cost policy so we measure the *environment*
ceiling, not policy inference. Two policy modes:
  --policy first   pick the first k legal options       (cheapest; pure env)
  --policy random  random.sample of k legal options     (adds RNG overhead)
And an optional --parse flag that runs `to_observation_class()` every decision,
to price the dataclass-parsing overhead the real RL loop will pay.

See rl_research/SELFPLAY_RL_PLAN.md (Phase 0). Run from the repo root:

    uv run python scripts/bench_throughput.py
    uv run python scripts/bench_throughput.py --procs 1,2,4,8,16 --duration 20
    uv run python scripts/bench_throughput.py --policy random --parse

Note: env binary is x86-64 only (agent/cg/libcg.so) — must run on an x86 host.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO / "agent"


def _load_deck() -> list[int]:
    lines = (AGENT_DIR / "deck.csv").read_text().split("\n")
    return [int(lines[i]) for i in range(60)]


def _legal_k(sel: dict) -> tuple[int, int]:
    """Mirror scripts/local_selfplay.legal_k on the raw select dict."""
    n = len(sel["option"])
    k = min(sel["maxCount"], n)
    if k < sel["minCount"]:
        k = min(sel["minCount"], n)
    return max(1, k), n


def _worker(args: tuple[int, float, int, str, bool, int]) -> dict:
    """Run self-play games for `duration` seconds; return throughput counters.

    Imports cg *inside* the process so each worker owns its own engine singleton
    (we never import cg in the parent — that would share the ctypes handle across
    a fork).
    """
    worker_id, duration, max_steps, policy_name, do_parse, seed = args
    os.chdir(AGENT_DIR)  # so libcg.so + deck.csv resolve as on Kaggle
    sys.path.insert(0, str(AGENT_DIR))
    rng = random.Random(seed)

    from cg.api import to_observation_class  # type: ignore[reportMissingImports]
    from cg.game import (  # type: ignore[reportMissingImports]
        battle_finish,
        battle_select,
        battle_start,
    )

    deck = _load_deck()

    def choose(sel: dict) -> list[int]:
        if sel is None:
            return deck
        k, n = _legal_k(sel)
        if policy_name == "random":
            return rng.sample(range(n), min(k, n))
        return list(range(min(k, n)))  # "first"

    games = decisions = errors = 0
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end:
        obs = None
        try:
            obs, start = battle_start(deck, deck)
            if obs is None:
                errors += 1
                continue
            for _ in range(max_steps):
                if do_parse:
                    oc = to_observation_class(obs)
                    res = oc.current.result if oc.current is not None else -1
                    sel = obs["select"]  # policy still acts on the raw dict
                else:
                    cur = obs.get("current")
                    res = cur.get("result", -1) if cur else -1
                    sel = obs["select"]
                if res is not None and res >= 0:
                    break
                obs = battle_select(choose(sel))
                decisions += 1
            games += 1
        except Exception:  # noqa: BLE001
            errors += 1
        finally:
            if obs is not None:
                with contextlib.suppress(Exception):
                    battle_finish()
    return {"games": games, "decisions": decisions, "errors": errors}


def run_grid(procs: list[int], duration: float, max_steps: int, policy: str, parse: bool) -> None:
    ctx = mp.get_context("spawn")  # fresh import of cg per worker; no inherited handle
    print(
        f"cabt throughput  policy={policy}  parse={parse}  duration={duration}s/worker  "
        f"host_cpus={os.cpu_count()}\n"
    )
    header = f"{'procs':>6} {'games/s':>10} {'decisions/s':>13} {'dec/game':>9} {'errors':>7}"
    print(header)
    print("-" * len(header))
    best = (0, 0.0)
    for p in procs:
        jobs = [(i, duration, max_steps, policy, parse, 1000 + i) for i in range(p)]
        t0 = time.perf_counter()
        with ctx.Pool(p) as pool:
            results = pool.map(_worker, jobs)
        wall = time.perf_counter() - t0
        g = sum(r["games"] for r in results)
        d = sum(r["decisions"] for r in results)
        e = sum(r["errors"] for r in results)
        gps, dps = g / wall, d / wall
        dpg = (d / g) if g else 0.0
        print(f"{p:>6} {gps:>10.1f} {dps:>13.0f} {dpg:>9.1f} {e:>7}")
        if dps > best[1]:
            best = (p, dps)
    print(
        f"\nPeak: {best[1]:,.0f} decisions/s at {best[0]} workers"
        f"  (~{best[1] * 3600 / 1e6:.1f}M decisions/hour, ~{best[1] * 86400 / 1e6:.0f}M/day)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", default="1,2,4,8", help="comma-separated worker counts to sweep")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds each worker runs")
    ap.add_argument("--max-steps", type=int, default=4000, help="safety cap on decisions/game")
    ap.add_argument("--policy", choices=["first", "random"], default="first")
    ap.add_argument("--parse", action="store_true", help="also run to_observation_class() per step")
    a = ap.parse_args()
    procs = [int(x) for x in a.procs.split(",") if x.strip()]
    run_grid(procs, a.duration, a.max_steps, a.policy, a.parse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
