"""Distributed self-play rollout collector (Phase 3, P3.1).

The throughput engine. The single-process `collect_rollout` in `ppo.py` encodes
and infers one decision at a time — fine for correctness, useless for scale. Here
a **persistent pool of W env-worker processes** (each owns one `battle_ptr`, pure
engine steppers + the torch-free encoder, *no torch* — see `dist_worker.py`) feeds
a **central batched inference loop** that holds the GPU model:

    workers: step engine → encode raw obs → send the decision request → block
    central: drain all pending requests → one batched `model.act()` on the GPU →
             scatter the chosen actions back → workers apply and continue

This exploits the Phase-0 L4 finding (`PHASE0_THROUGHPUT.md` Run 4): GPU inference
at our scale is overhead-bound and needs **batch ≥48 across rollout workers** to
amortize. With W workers, the natural inference batch is "however many workers sit
at a decision point right now" — so size W to push that batch past ~48.

Architecture choice — **the central process owns all RL state**. Workers ship one
`EncodedObs` per decision and get back a bare action list; they never touch
log-probs, values, GAE, or rewards. The central loop buckets each decision into
the right per-game / per-seat trajectory, and on game end computes the terminal
reward and hands the finished trajectories to the *same* `build_buffer_from_
trajectories` the single-process collector uses — so GAE and buffer layout are
identical by construction (the parity guarantee; see `tests/test_dist_collector`).

Self-play (`opponent="self"`) trains both seats with the current policy; a fixed
spec (`random`/`first`/`heuristic`) is stepped locally in the worker and only the
model seat's decisions are buffered. Seats are side-swapped per game via the token
the central loop dispatches.

The engine is a global singleton (`Battle.battle_ptr`) and unseedable, so this is
strictly process-level parallelism with one battle per worker; nothing is shared
but the queues. Workers persist across training iterations — only the model in the
central process changes between iters, and workers never hold it, so there is no
weight-sync step.
"""

from __future__ import annotations

import queue as queue_mod

import torch

from .dist_worker import DECIDE, PLAY, STOP, worker_main
from .encoding import EncodedObs
from .ppo import TrajStep, build_buffer_from_trajectories


class DistributedCollector:
    """A persistent pool of self-play env workers driven by one batched GPU loop.

    Build once, call `collect(model, n_games, ...)` each training iteration, then
    `close()`. Reuse across iterations is the point: spawning is expensive and the
    workers hold no model, so only the parent's `model` changes between calls.
    """

    def __init__(
        self,
        deck: list[int],
        *,
        n_workers: int = 8,
        opponent: str = "self",
        max_steps: int = 4000,
    ) -> None:
        self.deck = deck
        self.n_workers = n_workers
        self.opponent = opponent
        self.max_steps = max_steps
        self._ctx = torch.multiprocessing.get_context("spawn")
        self._task_q = self._ctx.Queue()
        self._req_q = self._ctx.Queue()
        self._resp_qs = [self._ctx.Queue() for _ in range(n_workers)]
        self._procs = []
        for wid in range(n_workers):
            p = self._ctx.Process(
                target=worker_main,
                args=(
                    wid,
                    opponent,
                    deck,
                    max_steps,
                    self._task_q,
                    self._req_q,
                    self._resp_qs[wid],
                ),
                daemon=True,
            )
            p.start()
            self._procs.append(p)
        self._closed = False

    def collect(
        self,
        model,
        n_games: int,
        *,
        gamma: float = 0.997,
        lam: float = 0.95,
        device: str = "cpu",
        seed: int = 0,
        batch_max: int = 4096,
    ):
        """Play `n_games` self-play games and return a GAE'd `RolloutBuffer`.

        Drop-in for `ppo.collect_rollout` — same buffer, same per-player GAE — but
        the decisions are batched across the worker pool through one GPU forward at
        a time. Blocks until all `n_games` finish.
        """
        if self._closed:
            raise RuntimeError("collector is closed")
        model.eval()
        gen = torch.Generator(device=device).manual_seed(seed)

        # Dispatch exactly n_games tokens; side-swap the model seat per game.
        for gidx in range(n_games):
            self._task_q.put((PLAY, gidx, gidx % 2, seed * 100003 + gidx))

        # Per-worker, per-seat in-progress trajectories. A worker plays one game at
        # a time, so (wid, seat) is unambiguous and reset on that worker's `done`.
        traj: list[dict[int, list[TrajStep]]] = [{0: [], 1: []} for _ in range(self.n_workers)]
        finished: list[tuple[list[TrajStep], float]] = []
        n_done = 0

        while n_done < n_games:
            decides: list[tuple[int, int, EncodedObs]] = []
            batch = [self._req_q.get()]  # block for at least one message
            while len(batch) < batch_max:  # then drain whatever else is waiting
                try:
                    batch.append(self._req_q.get_nowait())
                except queue_mod.Empty:
                    break

            for m in batch:
                if m[0] == DECIDE:
                    _, wid, seat, enc = m
                    decides.append((wid, seat, enc))
                else:  # DONE
                    _, wid, result = m
                    if result is not None:
                        for seat, steps in traj[wid].items():
                            if steps:
                                reward = 0.0 if result == 2 else (1.0 if result == seat else -1.0)
                                finished.append((steps, reward))
                    traj[wid] = {0: [], 1: []}
                    n_done += 1

            if decides:
                encs = [d[2] for d in decides]
                out = model.act(encs, sample=True, device=device, generator=gen)
                for (wid, seat, _enc), o in zip(decides, out, strict=True):
                    traj[wid][seat].append((_enc, o["action"], o["log_prob"], o["value"]))
                    self._resp_qs[wid].put(o["action"])

        return build_buffer_from_trajectories(finished, gamma, lam)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._procs:
            self._task_q.put((STOP,))
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> DistributedCollector:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
