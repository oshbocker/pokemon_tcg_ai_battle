"""Single-process self-play rollout + PPO (Phase 2, P2.5).

The end-to-end sanity loop: collect self-play games with the current policy, fit
with clipped PPO, and check the win rate against the honest baselines climbs. This
is the *correctness* loop before any scale — it's deliberately single-process
(one decision encoded/acted at a time). Phase 3 (P3.1) replaces the collector with
a batched multi-process one; nothing else here should need to change.

Engine notes: the cabt engine is a global singleton, so this runs one battle at a
time in-process. `libcg.so` resolves by absolute path, but the heuristic opponent
(`agent/main.py`) reads `deck.csv` relative to cwd, so the game loops chdir into
`agent/` and restore cwd in `finally`. Checkpoint paths in the train script are
absolute, so the chdir doesn't leak.
"""

from __future__ import annotations

import contextlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .encoding import EncodedObs, encode_observation
from .model import PtcgNet, collate

REPO = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO / "agent"


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    encoded: list[EncodedObs]
    actions: list[list[int]]
    logp: torch.Tensor  # [N] old log-probs (from the behaviour policy)
    value: torch.Tensor  # [N] old value estimates
    adv: torch.Tensor  # [N] GAE advantages
    ret: torch.Tensor  # [N] returns (adv + value)

    def __len__(self) -> int:
        return len(self.encoded)


def gae_terminal(
    values: list[float], terminal_reward: float, gamma: float, lam: float
) -> tuple[list[float], list[float]]:
    """GAE for one player's decision trajectory with a single terminal reward.

    The reward lands on the transition out of the last decision (next value = 0,
    nonterminal mask = 0 there); all earlier per-step rewards are 0."""
    n = len(values)
    adv = [0.0] * n
    last = 0.0
    for t in reversed(range(n)):
        nonterm = 0.0 if t == n - 1 else 1.0
        next_v = 0.0 if t == n - 1 else values[t + 1]
        reward = terminal_reward if t == n - 1 else 0.0
        delta = reward + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    ret = [adv[t] + values[t] for t in range(n)]
    return adv, ret


# ---------------------------------------------------------------------------
# Engine plumbing
# ---------------------------------------------------------------------------
def _import_engine():
    sys.path.insert(0, str(AGENT_DIR))
    from cg.api import to_observation_class  # type: ignore[reportMissingImports]
    from cg.game import (  # type: ignore[reportMissingImports]
        battle_finish,
        battle_select,
        battle_start,
    )

    return to_observation_class, battle_start, battle_select, battle_finish


def _make_fixed_opponent(spec: str, deck: list[int], rng: random.Random):
    """A non-trained opponent policy: 'random', 'first', or 'heuristic'."""
    to_oc, _, _, _ = _import_engine()
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


# ---------------------------------------------------------------------------
# Self-play rollout collection
# ---------------------------------------------------------------------------
def collect_rollout(
    model: PtcgNet,
    deck: list[int],
    n_games: int,
    *,
    opponent: str = "self",
    gamma: float = 0.997,
    lam: float = 0.95,
    device: str = "cpu",
    seed: int = 0,
    max_steps: int = 4000,
) -> RolloutBuffer:
    """Play `n_games` and return a flattened, GAE'd buffer of the model's decisions.

    `opponent='self'` is true self-play (both seats are the model; both seats'
    trajectories are trained). A fixed spec ('random'/'first'/'heuristic') trains
    only the model's seat — the simplest way to validate the loop learns to beat a
    known-weak baseline. Seats are swapped each game to cancel first-player bias.
    """
    model.eval()
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    prev_cwd = Path.cwd()
    os.chdir(AGENT_DIR)
    try:
        to_oc, battle_start, battle_select, battle_finish = _import_engine()
        opp_fn = None if opponent == "self" else _make_fixed_opponent(opponent, deck, rng)

        enc_all: list[EncodedObs] = []
        act_all: list[list[int]] = []
        logp_all: list[float] = []
        val_all: list[float] = []
        adv_all: list[float] = []
        ret_all: list[float] = []

        for g in range(n_games):
            model_seat = g % 2  # side-swap
            # per-player trajectory: list of (EncodedObs, action, logp, value)
            traj: dict[int, list[tuple]] = {0: [], 1: []}
            obs = None
            result = None
            try:
                obs, _ = battle_start(deck, deck)
                if obs is None:
                    continue
                for _ in range(max_steps):
                    oc = to_oc(obs)
                    cur = oc.current
                    if cur is None:
                        break
                    if cur.result is not None and cur.result >= 0:
                        result = cur.result
                        break
                    seat = cur.yourIndex
                    if opp_fn is not None and seat != model_seat:  # fixed opponent's turn
                        obs = battle_select(opp_fn(obs))
                        continue
                    enc = encode_observation(obs)
                    out = model.act([enc], sample=True, device=device, generator=gen)[0]
                    traj[seat].append((enc, out["action"], out["log_prob"], out["value"]))
                    obs = battle_select(out["action"])
            finally:
                if obs is not None:
                    with contextlib.suppress(Exception):
                        battle_finish()
            if result is None:
                continue

            seats = (0, 1) if opponent == "self" else (model_seat,)
            for seat in seats:
                steps = traj[seat]
                if not steps:
                    continue
                reward = 0.0 if result == 2 else (1.0 if result == seat else -1.0)
                values = [s[3] for s in steps]
                adv, ret = gae_terminal(values, reward, gamma, lam)
                for k, (enc, action, logp, value) in enumerate(steps):
                    enc_all.append(enc)
                    act_all.append(action)
                    logp_all.append(logp)
                    val_all.append(value)
                    adv_all.append(adv[k])
                    ret_all.append(ret[k])
    finally:
        os.chdir(prev_cwd)

    return RolloutBuffer(
        encoded=enc_all,
        actions=act_all,
        logp=torch.tensor(logp_all, dtype=torch.float32),
        value=torch.tensor(val_all, dtype=torch.float32),
        adv=torch.tensor(adv_all, dtype=torch.float32),
        ret=torch.tensor(ret_all, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    epochs: int = 4
    minibatch: int = 256
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4


def ppo_update(
    model: PtcgNet,
    optimizer: torch.optim.Optimizer,
    buf: RolloutBuffer,
    cfg: PPOConfig,
    device: str = "cpu",
) -> dict:
    """One PPO pass over `buf`. Returns scalar metrics for logging."""
    model.train()
    n = len(buf)
    adv = buf.adv.to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise advantages
    ret = buf.ret.to(device)
    old_logp = buf.logp.to(device)
    old_val = buf.value.to(device)

    metrics = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
    n_batches = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch].tolist()
            batch = collate([buf.encoded[i] for i in idx], device)
            actions = [buf.actions[i] for i in idx]
            new_logp, entropy, value = model.evaluate_actions(batch, actions)

            ratio = (new_logp - old_logp[idx]).exp()
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            # clipped value loss (PPO2 style)
            v_clip = old_val[idx] + (value - old_val[idx]).clamp(-cfg.clip, cfg.clip)
            vf = 0.5 * torch.max((value - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
            ent = entropy.mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["pg_loss"] += float(pg)
                metrics["vf_loss"] += float(vf)
                metrics["entropy"] += float(ent)
                metrics["approx_kl"] += float((old_logp[idx] - new_logp).mean())
                metrics["clipfrac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
            n_batches += 1

    return {k: v / max(1, n_batches) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# In-process evaluation (quick win-rate vs a fixed opponent)
# ---------------------------------------------------------------------------
@torch.no_grad()
def quick_eval(
    model: PtcgNet,
    deck: list[int],
    opponent: str,
    n_games: int,
    *,
    device: str = "cpu",
    seed: int = 12345,
    max_steps: int = 4000,
) -> dict:
    """Greedy model vs a fixed opponent, side-swapped. Returns win/loss/draw + rate.

    Lower-variance than the training rollout (greedy), but still a *small-n* gut
    check — trust `scripts/eval.py` (high-n, Wilson CIs) for real decisions."""
    model.eval()
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    prev_cwd = Path.cwd()
    os.chdir(AGENT_DIR)
    wins = losses = draws = 0
    try:
        to_oc, battle_start, battle_select, battle_finish = _import_engine()
        opp_fn = _make_fixed_opponent(opponent, deck, rng)
        for g in range(n_games):
            model_seat = g % 2
            obs = None
            result = None
            try:
                obs, _ = battle_start(deck, deck)
                if obs is None:
                    continue
                for _ in range(max_steps):
                    oc = to_oc(obs)
                    cur = oc.current
                    if cur is None:
                        break
                    if cur.result is not None and cur.result >= 0:
                        result = cur.result
                        break
                    if cur.yourIndex == model_seat:
                        enc = encode_observation(obs)
                        out = model.act([enc], sample=False, device=device, generator=gen)[0]
                        obs = battle_select(out["action"])
                    else:
                        obs = battle_select(opp_fn(obs))
            finally:
                if obs is not None:
                    with contextlib.suppress(Exception):
                        battle_finish()
            if result is None:
                continue
            if result == 2:
                draws += 1
            elif result == model_seat:
                wins += 1
            else:
                losses += 1
    finally:
        os.chdir(prev_cwd)
    decisive = wins + losses
    return {
        "opponent": opponent,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": (wins / decisive) if decisive else float("nan"),
        "n": decisive,
    }


def load_deck(path: Path | None = None) -> list[int]:
    p = path or (AGENT_DIR / "deck.csv")
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    return [int(lines[i]) for i in range(60)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
