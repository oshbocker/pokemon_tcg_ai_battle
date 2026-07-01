"""Distributed self-play / league rollout collector (Phase 3, P3.1 + P3.4).

The throughput engine. A **persistent pool of W env-worker processes** (each owns
one `battle_ptr`, pure engine steppers + the torch-free encoder, *no torch* — see
`dist_worker.py`) feeds a **central batched inference loop** that holds the GPU
model(s):

    workers: step engine → encode raw obs → send the decision request → block
    central: drain all pending requests → batched `model.act()` on the GPU →
             scatter the chosen actions back → workers apply and continue

This exploits the L4 finding (`PHASE0_THROUGHPUT.md`): GPU inference is overhead-
bound at our scale and needs **batch ≥48 across rollout workers** to amortize.

**League (P3.4).** Pure self-play collapses into a degenerate mutual-best-response
(entropy → 0, sub-random play). The fix is to face the trainee with a *mix* of
opponents, sampled per game from a `League`:

  * ``"self"``        — current policy on both seats (both trained);
  * ``"model:<id>"``  — a frozen **past checkpoint** held in `League.models`, run
                       on the GPU here but **not** trained;
  * fixed agents      — ``"random"`` / ``"heuristic"`` / ``"kaggle:<name>"`` …,
                       stepped locally in the worker.

**Only the current policy's decisions are trained.** The central loop tags every
decision with a policy id (``"cur"`` vs ``"model:<id>"``), batches per policy, runs
each through its net, and buffers *only* the ``"cur"`` decisions — so a `model:`
opponent contributes pressure but no gradient. Trajectories/GAE flow through the
same `build_buffer_from_trajectories` as the single-process collector (parity by
construction). Workers persist across iterations and hold no model, so only the
parent's `model` + frozen pool change between calls — no weight-sync step.
"""

from __future__ import annotations

import queue as queue_mod
import random
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .dist_worker import CUR, DECIDE, PLAY, STOP, worker_main
from .encoding import EncodedObs
from .model import PtcgNet
from .ppo import TrajStep, build_buffer_from_trajectories


@dataclass
class League:
    """Opponent pool sampled per game. `mix` is `[(spec, weight), ...]` over
    `"self"`, `"model:<id>"` (ids must key `models`), and fixed-agent specs.

    `decks` maps a spec → the deck that opponent pilots (asymmetric self-play): a
    Kaggle/fixed opponent plays ITS OWN archetype deck, while ``self``/``model:``
    opponents are absent from `decks` and default to the trainee deck (a mirror).
    Decks are data — the collector ships `decks.get(spec)` in the PLAY token, so
    swapping an opponent's deck never touches the worker/collector code."""

    mix: list[tuple[str, float]] = field(default_factory=lambda: [("self", 1.0)])
    models: dict[str, PtcgNet] = field(default_factory=dict)  # frozen past checkpoints
    decks: dict[str, list[int]] = field(default_factory=dict)  # spec -> opponent deck

    def fixed_specs(self) -> list[str]:
        """The locally-stepped (non-model) opponent specs this league can request."""
        return [s for s, _ in self.mix if s != "self" and not s.startswith("model:")]


_SELF_PLAY = League()


def league_fixed_specs(
    *,
    w_heuristic: float,
    w_random: float,
    kaggle: str | None,
    w_kaggle: float,
    extra: list[tuple[str, float]] | None = None,
) -> list[str]:
    """The locally-stepped opponent specs a league config will request (stable across
    iterations — pass to the collector constructor so workers prebuild these agents).

    `extra` folds in the manifest opponents (`(spec, weight)`); only the locally-
    stepped ones (`heuristic`/`random`/`first`/`kaggle:*`, weight > 0) are returned —
    `self`/`model:` opponents run on the central GPU, not in the worker."""
    specs: list[str] = []
    if w_heuristic > 0:
        specs.append("heuristic")
    if w_random > 0:
        specs.append("random")
    if kaggle and w_kaggle > 0:
        specs.append(f"kaggle:{kaggle}")
    for spec, w in extra or []:
        if w > 0 and spec != "self" and not spec.startswith("model:"):
            specs.append(spec)
    return list(dict.fromkeys(specs))  # dedupe, preserve order


def build_league(
    pool: dict[str, PtcgNet],
    *,
    w_self: float = 1.0,
    w_past: float = 2.0,
    w_heuristic: float = 1.0,
    w_random: float = 0.5,
    kaggle: str | None = None,
    w_kaggle: float = 1.0,
    extra: list[tuple[str, float]] | None = None,
    opp_decks: dict[str, list[int]] | None = None,
    ext_models: dict[str, PtcgNet] | None = None,
) -> League:
    """Assemble a `League` from the current past-checkpoint `pool` + fixed agents.

    `w_past` is the *total* weight for past checkpoints, split evenly across the
    pool, so the per-checkpoint weight shrinks as the pool grows (the aggregate
    pressure from "an older me" stays constant). Specs with weight 0 are omitted.

    `extra` is the manifest-driven opponent set — additional `(spec, weight)` pairs
    (e.g. `("kaggle:archaludon", 1.0)`, `("heuristic", 0.5)`) that, with `opp_decks`
    (spec → that opponent's own deck), make the league a mixed, asymmetric pool. It
    coexists with the legacy single-`kaggle` flag; both feed the same `decks` map.

    `ext_models` are *external* frozen checkpoints (id → net) — e.g. a trained
    Archaludon `best.pt` dropped into the Alakazam league. Unlike the trainee's own
    `pool` snapshots (which mirror the trainee deck and get evicted), these are
    never evicted and pilot their own deck (supplied via `extra`/`opp_decks` as a
    `("model:<id>", weight)` entry). They are merged into `League.models` so the
    central GPU loop resolves them by id, exactly like a past-self checkpoint."""
    mix: list[tuple[str, float]] = [("self", w_self)]
    if pool and w_past > 0:
        per = w_past / len(pool)
        mix.extend((f"model:{pid}", per) for pid in pool)
    if w_heuristic > 0:
        mix.append(("heuristic", w_heuristic))
    if w_random > 0:
        mix.append(("random", w_random))
    if kaggle and w_kaggle > 0:
        mix.append((f"kaggle:{kaggle}", w_kaggle))
    if extra:
        mix.extend((spec, w) for spec, w in extra if w > 0)
    # Guard a degenerate all-zero distribution (e.g. --w-self 0 with an empty pool and
    # every other weight 0): the per-game sampler needs a positive total, so fall back
    # to pure self-play rather than crash collect() with a cryptic ValueError.
    if sum(w for _, w in mix) <= 0:
        mix = [("self", 1.0)]
    models = pool if not ext_models else {**pool, **ext_models}
    return League(mix=mix, models=models, decks=dict(opp_decks or {}))


class DistributedCollector:
    """A persistent pool of env workers driven by one batched GPU loop.

    Build once with the *fixed* opponent specs the workers may ever play locally
    (so they can prebuild those agents), then call `collect(model, n_games,
    league=...)` each iteration and `close()` at the end.
    """

    def __init__(
        self,
        deck: list[int],
        *,
        n_workers: int = 8,
        fixed_specs: tuple[str, ...] | list[str] = (),
        max_steps: int = 4000,
    ) -> None:
        self.deck = deck
        self.n_workers = n_workers
        self.fixed_specs = list(fixed_specs)
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
                    self.fixed_specs,
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
        model: PtcgNet,
        n_games: int,
        *,
        league: League | None = None,
        gamma: float = 0.997,
        lam: float = 0.95,
        device: str = "cpu",
        seed: int = 0,
        batch_max: int = 4096,
    ):
        """Play `n_games` and return a GAE'd `RolloutBuffer` of the *current* policy's
        decisions. `league=None` is pure self-play. Drop-in for `ppo.collect_rollout`.
        """
        if self._closed:
            raise RuntimeError("collector is closed")
        league = league or _SELF_PLAY
        model.eval()
        for fm in league.models.values():
            fm.eval()
        gen = torch.Generator(device=device).manual_seed(seed)
        spec_rng = random.Random(seed ^ 0x9E3779B9)
        specs, weights = zip(*league.mix, strict=True)

        # Dispatch exactly n_games tokens; side-swap the model seat, sample opponent,
        # and ship that opponent's own deck (None = mirror the trainee deck).
        for gidx in range(n_games):
            opp_spec = spec_rng.choices(specs, weights=weights, k=1)[0]
            opp_deck = league.decks.get(opp_spec)
            self._task_q.put((PLAY, gidx, gidx % 2, seed * 100003 + gidx, opp_spec, opp_deck))

        # Per-worker, per-seat in-progress trajectories — only the CURRENT policy's
        # ("cur") decisions are buffered; frozen-opponent decisions are inference-only.
        traj: list[dict[int, list[TrajStep]]] = [{0: [], 1: []} for _ in range(self.n_workers)]
        finished: list[tuple[list[TrajStep], float]] = []
        n_done = 0

        while n_done < n_games:
            batch = [self._req_q.get()]  # block for at least one message
            while len(batch) < batch_max:  # then drain whatever else is waiting
                try:
                    batch.append(self._req_q.get_nowait())
                except queue_mod.Empty:
                    break

            decides: dict[str, list[tuple[int, int, EncodedObs]]] = defaultdict(list)
            for m in batch:
                if m[0] == DECIDE:
                    _, wid, seat, policy, enc = m
                    decides[policy].append((wid, seat, enc))
                else:  # DONE
                    _, wid, result = m
                    if result is not None:
                        for seat, steps in traj[wid].items():
                            if steps:
                                reward = 0.0 if result == 2 else (1.0 if result == seat else -1.0)
                                finished.append((steps, reward))
                    traj[wid] = {0: [], 1: []}
                    n_done += 1

            # One batched forward per distinct policy (current net + each frozen ckpt).
            for policy, items in decides.items():
                net = model if policy == CUR else league.models[policy[len("model:") :]]
                out = net.act([e for _, _, e in items], sample=True, device=device, generator=gen)
                for (wid, seat, enc), o in zip(items, out, strict=True):
                    if policy == CUR:  # only the trained policy contributes gradient
                        traj[wid][seat].append((enc, o["action"], o["log_prob"], o["value"]))
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
