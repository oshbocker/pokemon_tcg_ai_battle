"""High-n, side-swapped, unpaired evaluation harness (Phase 5 skeleton).

The cabt engine is **unseedable** (P0.4) — it draws its `mt19937` seed from
`std::random_device` per battle — so the Orbit Wars paired-seed / common-random-
number trick is unavailable. We compensate the way Phase 0 said we could:
**brute-force the variance with volume**. Win/loss is Bernoulli; a 5 pp edge
(55% vs 50%) needs ~1.5K games/arm at ~80% power, a 3 pp edge ~4.3K — cheap given
~180 engine-games/s on an L4. We always **side-swap seats** (champion plays seat
0 for half the games, seat 1 for the other half) to cancel the first-player bias,
which is independent of seeding.

Parallelism is **process-level** (the engine is a global singleton —
`Battle.battle_ptr`), so games run in `multiprocessing` worker chunks; `cg` is
imported *inside* the worker so each process owns its own engine handle. Results
stream to a **resumable CSV** (one row per finished chunk): re-running tops up to
the requested game count instead of starting over.

This module is engine-only at import (torch is imported lazily, only when a
`model:<path>` agent is resolved). Agents are named specs resolved per worker
(`heuristic`, `random`, `first`, `mirror`, `model:<path>`), so they pickle cleanly
across a spawn — the trained policy loads its own net per worker on CPU and acts
greedily. The heuristic agent (`agent/main.py`) keeps
**module-global** turn state, so each agent slot loads its *own* module instance —
otherwise champion and opponent would clobber each other's `plan` in a mirror.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import math
import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO / "agent"
KAGGLE_AGENT_DIR = AGENT_DIR / "kaggle_agents"

# Agent specs that don't need a path argument.
NAMED_AGENTS = ("heuristic", "random", "first", "mirror")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (p_hat, lo, hi). Far better than the normal approximation at the
    sample sizes and near-0/1 rates we hit early. `z=1.96` ≈ 95%.
    """
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def games_for_edge(edge_pp: float, base: float = 0.50, power: float = 0.80) -> int:
    """Approx games/arm to detect a `edge_pp`-point win-rate edge over `base` at
    two-sided α=0.05 and the given power (normal approximation). Planning aid for
    `--games`, and the basis for the 'don't act below this n' floor."""
    z_a, z_b = 1.959964, 0.841621 if power == 0.80 else _z_for_power(power)
    p1, p2 = base, base + edge_pp / 100.0
    pbar = (p1 + p2) / 2
    num = z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / (p2 - p1)) ** 2)


def _z_for_power(power: float) -> float:
    # Inverse normal CDF via a rational approximation (Acklam); good to ~1e-9.
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    p = power
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


# ---------------------------------------------------------------------------
# Agent construction (inside the worker process)
# ---------------------------------------------------------------------------
def _load_main_module(name: str):
    """Load agent/main.py as a *fresh* module instance (own globals)."""
    spec = importlib.util.spec_from_file_location(name, AGENT_DIR / "main.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_deck_csv(path: str | Path | None = None) -> list[int]:
    """A 60-card deck CSV — the deck a `model:`/`random`/`first` champion plays.

    Defaults to `agent/deck.csv`; pass a path (e.g. `agent/decks/archaludon.csv`) to
    evaluate a model on a specific trainee deck. Relative paths resolve at the repo
    root (workers chdir into `agent/`)."""
    p = Path(path) if path else (AGENT_DIR / "deck.csv")
    if not p.is_absolute():
        p = REPO / p
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    return [int(lines[i]) for i in range(60)]


def _build_model_agent(spec: str, device: str = "cpu", deck_path: str | None = None):
    """Resolve `model:<path>` to (greedy agent_fn, deck). Imports torch lazily so
    the engine-only specs (and the encoding tests) never pull in the rl extra.

    The checkpoint is the `{model, cfg}` dict written by `scripts/train_selfplay.py`
    / the ablation. The agent is **greedy** (`sample=False`) — evaluation wants the
    policy's mode, not exploration. Path is resolved relative to the repo root when
    not absolute (workers chdir into `agent/`, so a bare path would break)."""
    import torch  # noqa: PLC0415 — lazy: keep torch out of the engine-only path

    from .encoding import encode_observation
    from .model import ModelConfig, PtcgNet

    raw = spec[len("model:") :]
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["cfg"]) if isinstance(ckpt.get("cfg"), dict) else ModelConfig()
    model = PtcgNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    deck = _load_deck_csv(deck_path)

    @torch.no_grad()
    def agent_fn(obs_dict):
        if obs_dict.get("select") is None:
            return deck
        out = model.act([encode_observation(obs_dict)], sample=False, device=device)
        return out[0]["action"]

    return agent_fn, deck


def _build_agent(
    spec: str, deck: list[int], rng: random.Random, slot: str, deck_path: str | None = None
):
    """Resolve an agent spec to (agent_fn, deck). Called once per worker, per slot.

    `deck_path` is the champion's chosen deck for the deck-carrying-less specs
    (`model:`/`random`/`first`); heuristic/mirror/kaggle agents ignore it and pilot
    their own `my_deck` (their heuristics are archetype-specific)."""
    from cg.api import to_observation_class  # type: ignore[reportMissingImports]

    if spec in ("heuristic", "mirror"):
        mod = _load_main_module(f"agent_{slot}")
        return mod.agent, mod.my_deck
    if spec in ("random", "first"):

        def agent_fn(obs_dict, _spec=spec):
            oc = to_observation_class(obs_dict)
            if oc.select is None:
                return deck
            n = len(oc.select.option)
            k = min(oc.select.maxCount, n)
            if k < oc.select.minCount:
                k = min(oc.select.minCount, n)
            k = max(1, k)
            if _spec == "random":
                return rng.sample(range(n), min(k, n))
            return list(range(min(k, n)))

        return agent_fn, deck
    if spec.startswith("model:"):
        return _build_model_agent(spec, deck_path=deck_path)
    if spec.startswith("kaggle:"):
        name = spec[len("kaggle:") :]
        s = importlib.util.spec_from_file_location(
            f"eval_{slot}_{name}", KAGGLE_AGENT_DIR / f"{name}.py"
        )
        assert s is not None and s.loader is not None
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod.agent, mod.my_deck  # the borrowed agent pilots its own deck
    raise ValueError(f"unknown agent spec: {spec!r}")


# ---------------------------------------------------------------------------
# Worker: play a chunk of games, champion fixed to one seat
# ---------------------------------------------------------------------------
@dataclass
class ChunkResult:
    opponent: str
    champion: str
    champ_seat: int
    games: int
    champ_wins: int
    champ_losses: int
    draws: int
    errors: int


def _worker_chunk(task: tuple) -> ChunkResult:
    champion_spec, opponent_spec, n_games, champ_seat, base_seed, max_steps, champion_deck = task
    os.chdir(AGENT_DIR)
    sys.path.insert(0, str(AGENT_DIR))
    rng = random.Random(base_seed)

    from cg.api import to_observation_class  # type: ignore[reportMissingImports]
    from cg.game import (  # type: ignore[reportMissingImports]
        battle_finish,
        battle_select,
        battle_start,
    )

    # The champion's chosen deck (for model:/random/first); mirror copies the champion.
    champ_override = _load_deck_csv(champion_deck) if champion_deck else []
    opp_resolved = champion_spec if opponent_spec == "mirror" else opponent_spec
    champ_fn, champ_deck = _build_agent(
        champion_spec, champ_override, rng, slot="champ", deck_path=champion_deck
    )
    # A mirror opponent is the champion's policy on the champion's deck; any other
    # opponent pilots its own deck (kaggle/heuristic) or borrows champ_deck (random).
    opp_deck_path = champion_deck if opponent_spec == "mirror" else None
    opp_fn, opp_deck = _build_agent(
        opp_resolved, champ_deck, rng, slot="opp", deck_path=opp_deck_path
    )
    if not champ_deck:  # random/first champion: borrow the heuristic deck
        champ_deck = opp_deck if opp_deck else _load_main_module("agent_deck").my_deck

    # Seat the champion; the other seat is the opponent (build tuples directly so
    # the slots are typed as callables/decks, not list[None]).
    opp_deck = opp_deck or champ_deck
    if champ_seat == 0:
        seat_fn = (champ_fn, opp_fn)
        seat_deck = (champ_deck, opp_deck)
    else:
        seat_fn = (opp_fn, champ_fn)
        seat_deck = (opp_deck, champ_deck)

    wins = losses = draws = errors = 0
    for _ in range(n_games):
        obs = None
        try:
            obs, start = battle_start(seat_deck[0], seat_deck[1])
            if obs is None:
                errors += 1
                continue
            res = None
            for _ in range(max_steps):
                oc = to_observation_class(obs)
                cur = oc.current
                if cur is None:
                    break
                if cur.result is not None and cur.result >= 0:
                    res = cur.result
                    break
                obs = battle_select(seat_fn[cur.yourIndex](obs))
            if res is None:
                errors += 1
            elif res == 2:
                draws += 1
            elif res == champ_seat:
                wins += 1
            else:
                losses += 1
        except Exception:  # noqa: BLE001 — a crash is a forfeit, not a harness stop
            errors += 1
        finally:
            if obs is not None:
                with contextlib.suppress(Exception):
                    battle_finish()
    return ChunkResult(
        opponent=opponent_spec,
        champion=champion_spec,
        champ_seat=champ_seat,
        games=n_games,
        champ_wins=wins,
        champ_losses=losses,
        draws=draws,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "opponent",
    "champion",
    "champ_seat",
    "games",
    "champ_wins",
    "champ_losses",
    "draws",
    "errors",
]


@dataclass
class MatchupSummary:
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    errors: int
    seat0_games: int
    seat1_games: int

    @property
    def decisive(self) -> int:
        return self.wins + self.losses

    def winrate_ci(self, z: float = 1.96) -> tuple[float, float, float]:
        """Win rate over decisive games (draws excluded), with a Wilson interval."""
        return wilson_interval(self.wins, self.decisive, z)


def _read_csv_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, rows: list[ChunkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in CSV_FIELDS})


def _summarise(rows: list[dict], opponent: str) -> MatchupSummary:
    s = MatchupSummary(opponent, 0, 0, 0, 0, 0, 0, 0)
    for r in rows:
        if r["opponent"] != opponent:
            continue
        g = int(r["games"])
        s.games += g
        s.wins += int(r["champ_wins"])
        s.losses += int(r["champ_losses"])
        s.draws += int(r["draws"])
        s.errors += int(r["errors"])
        if int(r["champ_seat"]) == 0:
            s.seat0_games += g
        else:
            s.seat1_games += g
    return s


def _init_eval_worker() -> None:
    """Pool initializer — pin each worker to a single thread.

    The honest eval runs `cpu_count-1` worker processes, each doing **batch-1 CPU
    inference** for `model:<path>` agents. Without this, every worker's torch/BLAS
    pool sizes itself to the whole machine (~one thread per core), so the workers
    oversubscribe the box by ~`cpu_count`× and grind to a near-stall — the cause of
    the option-rank A/B honest-eval "hang" (it crept across two days via the
    resumable CSV and never finished a session). Set the env vars (BLAS/OpenMP read
    them at import; a fresh spawn worker hasn't imported numpy yet) **and** torch's
    runtime thread count (guarded — engine-only matchups never import torch)."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"
    with contextlib.suppress(ImportError):
        import torch  # noqa: PLC0415 — lazy: keep torch out of the engine-only path

        torch.set_num_threads(1)


def evaluate(
    champion: str,
    opponents: list[str],
    games: int,
    out_csv: Path,
    workers: int = 0,
    max_steps: int = 4000,
    chunk: int = 25,
    base_seed: int = 1000,
    chunk_timeout: float = 1200.0,
    champion_deck: str | None = None,
) -> dict[str, MatchupSummary]:
    """Run (or resume) `games` side-swapped games of `champion` vs each opponent.

    `games` is the *total* per opponent, split 50/50 across the two seats. Resumes
    from whatever is already in `out_csv`, dispatching only the shortfall. Returns
    the per-opponent summary keyed by opponent name.

    Workers are single-threaded (`_init_eval_worker`) so the many CPU-inference
    processes don't oversubscribe the machine. `chunk_timeout` (s) bounds the wait
    on any one chunk: if a worker wedges — e.g. the native engine looping on a
    degenerate board, which `max_steps` (a Python guard) can't interrupt — the
    matchup is left partial (resumable) instead of hanging the whole run forever.
    """
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    ctx = mp.get_context("spawn")
    progress = _read_csv_progress(out_csv)

    for opp in opponents:
        done = _summarise(progress, opp)
        target_per_seat = games // 2
        tasks: list[tuple] = []
        seed = base_seed + abs(hash(opp)) % 100000
        for seat in (0, 1):
            have = done.seat0_games if seat == 0 else done.seat1_games
            need = max(0, target_per_seat - have)
            n_chunks = math.ceil(need / chunk) if need else 0
            for c in range(n_chunks):
                this = min(chunk, need - c * chunk)
                tasks.append(
                    (champion, opp, this, seat, seed + seat * 1000 + c, max_steps, champion_deck)
                )
        if not tasks:
            continue
        with ctx.Pool(workers, initializer=_init_eval_worker) as pool:
            pending = [pool.apply_async(_worker_chunk, (t,)) for t in tasks]
            for done_chunks, ar in enumerate(pending):
                try:
                    res = ar.get(timeout=chunk_timeout)
                except mp.TimeoutError:
                    pool.terminate()
                    print(
                        f"  [eval] WEDGED: chunk for {champion} vs {opp} exceeded "
                        f"{chunk_timeout:.0f}s; abandoning matchup at {done_chunks}/"
                        f"{len(tasks)} chunks (partial CSV is resumable).",
                        flush=True,
                    )
                    break
                _append_csv(out_csv, [res])

    final = _read_csv_progress(out_csv)
    return {opp: _summarise(final, opp) for opp in opponents}


def dont_act_floor(aa_summary: MatchupSummary, edge_pp: float = 5.0) -> dict:
    """From a measured A/A null (champion vs an identical policy), report the
    noise picture and the 'don't act below this n' floor.

    The A/A win rate should sit at ~50%; its Wilson half-width at the achieved n
    is the empirical noise. The floor is the games/arm at which a real `edge_pp`
    edge becomes statistically separable — below it, treat any 'win' as noise.
    """
    p, lo, hi = aa_summary.winrate_ci()
    return {
        "aa_winrate": p,
        "aa_ci": (lo, hi),
        "aa_halfwidth_pp": (hi - lo) / 2 * 100,
        "aa_decisive_games": aa_summary.decisive,
        "aa_draws": aa_summary.draws,
        "floor_games_per_arm_5pp": games_for_edge(5.0),
        "floor_games_per_arm_3pp": games_for_edge(3.0),
        "floor_games_per_arm_custom": games_for_edge(edge_pp),
        "edge_pp": edge_pp,
    }
