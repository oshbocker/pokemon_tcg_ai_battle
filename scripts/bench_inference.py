"""P0.3 — policy inference-cost probe.

Phase 0 measured the *environment* ceiling (engine-only ~25K dec/s on an L4, ~81K
on Blackwell). The missing number before we can set a model-size band is the
*policy* cost: how long a forward pass of the entity-token + pointer-head model
takes at realistic token counts, and whether that makes inference — rather than
the CPU env — the throughput bottleneck.

This builds the model in the shape `docs/rl-obs-action.md` §5 specifies (card/
attack/role embeddings → shared transformer trunk → pointer actor over option
tokens + a value head), at a few size bands, and times batched forward passes on
CPU (and CUDA if present). It then combines per-decision latency with the Phase-0
env numbers to report **policy-included decisions/s** and pick a first band.

    uv run --extra rl python scripts/bench_inference.py
    uv run --extra rl python scripts/bench_inference.py --sizes small,medium --batches 1,8,32

Token-count regimes come from real games (P1.4 fixture): entities ~10–20, options
1–34 typically, with a padded worst case up to ~1000 options. Batch = decisions
collected across rollout workers into one forward pass (12 on L4, 24–48 on
Blackwell). This is a *probe*: inputs are synthetic ints of the right shapes, so
it measures compute/latency, not policy quality.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptcg_battle.encoding import (  # noqa: E402
    ATTACK_VOCAB,
    CARD_VOCAB,
    ENTITY_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    N_AREA,
    N_ENERGY_TYPES,
    N_OPTION_TYPE,
    N_ROLE,
    N_SELECT_CONTEXT,
    N_SPECIAL_COND,
    OPTION_FEAT_DIM,
)

# Phase-0 measured engine-only throughput (decisions/s), for the combined report.
ENV_DEC_S = {"L4 (12 vCPU)": 25_000, "Blackwell (48 vCPU)": 81_000}


@dataclass
class ModelConfig:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int


SIZE_BANDS = {
    "tiny": ModelConfig("tiny", 128, 4, 4, 512),
    "small": ModelConfig("small", 256, 6, 8, 1024),
    "medium": ModelConfig("medium", 384, 8, 8, 1536),
    "large": ModelConfig("large", 512, 10, 8, 2048),
}


class PointerValueNet(nn.Module):
    """Entity+option tokens → shared transformer → pointer logits over options + value.

    Faithful to the §5 contract's shapes (so latency/params are representative);
    the parameter-heavy card/attack ID embeddings are included since they're real
    memory and a real bundle-size cost downstream."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.card_emb = nn.Embedding(CARD_VOCAB, d, padding_idx=0)
        self.attack_emb = nn.Embedding(ATTACK_VOCAB, d, padding_idx=0)
        self.role_emb = nn.Embedding(N_ROLE, d)
        self.area_emb = nn.Embedding(N_AREA, d)
        self.inplay_emb = nn.Embedding(N_AREA, d)
        self.opttype_emb = nn.Embedding(N_OPTION_TYPE, d)
        self.special_emb = nn.Embedding(N_SPECIAL_COND, d)
        self.ctx_emb = nn.Embedding(N_SELECT_CONTEXT, d)
        self.ent_feat = nn.Linear(ENTITY_FEAT_DIM + N_ENERGY_TYPES, d)
        self.opt_feat = nn.Linear(OPTION_FEAT_DIM, d)
        self.global_feat = nn.Linear(GLOBAL_FEAT_DIM, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.query = nn.Linear(d, d)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1), nn.Tanh())

    def forward(self, ent_tok, opt_tok, ctx, ent_x, opt_x, glob_x):
        # ent_tok/opt_tok: integer id stacks; *_x: float feature stacks. Shapes are
        # the only thing that matters for the probe; we sum a couple of embeddings.
        ent = self.card_emb(ent_tok) + self.role_emb(ent_x["role"]) + self.ent_feat(ent_x["feat"])
        opt = (
            self.card_emb(opt_tok)
            + self.attack_emb(opt_x["attack"])
            + self.opttype_emb(opt_x["type"])
            + self.area_emb(opt_x["area"])
            + self.inplay_emb(opt_x["inplay"])
            + self.special_emb(opt_x["special"])
            + self.opt_feat(opt_x["feat"])
        )
        ent = ent + self.global_feat(glob_x).unsqueeze(1)  # broadcast global onto entities
        n_opt = opt.shape[1]
        seq = torch.cat([ent, opt], dim=1)
        h = self.trunk(seq)
        opt_h = h[:, -n_opt:, :]
        q = self.query(h.mean(dim=1) + self.ctx_emb(ctx))  # [B, d]
        logits = torch.einsum("bld,bd->bl", opt_h, q)  # pointer scores over options
        value = self.value_head(h.mean(dim=1)).squeeze(-1)
        return logits, value


def _synthetic_batch(b: int, n_ent: int, n_opt: int, device) -> dict:
    g = torch.Generator().manual_seed(0)

    def ints(hi, *shape):
        return torch.randint(0, hi, shape, generator=g).to(device)

    return dict(
        ent_tok=ints(CARD_VOCAB, b, n_ent),
        opt_tok=ints(CARD_VOCAB, b, n_opt),
        ctx=ints(N_SELECT_CONTEXT, b),
        ent_x=dict(
            role=ints(N_ROLE, b, n_ent),
            feat=torch.rand(b, n_ent, ENTITY_FEAT_DIM + N_ENERGY_TYPES, generator=g).to(device),
        ),
        opt_x=dict(
            attack=ints(ATTACK_VOCAB, b, n_opt),
            type=ints(N_OPTION_TYPE, b, n_opt),
            area=ints(N_AREA, b, n_opt),
            inplay=ints(N_AREA, b, n_opt),
            special=ints(N_SPECIAL_COND, b, n_opt),
            feat=torch.rand(b, n_opt, OPTION_FEAT_DIM, generator=g).to(device),
        ),
        glob_x=torch.rand(b, GLOBAL_FEAT_DIM, generator=g).to(device),
    )


def _time_forward(model, batch, iters: int) -> float:
    """Median seconds per forward pass."""
    is_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(
                batch["ent_tok"],
                batch["opt_tok"],
                batch["ctx"],
                batch["ent_x"],
                batch["opt_x"],
                batch["glob_x"],
            )
        if is_cuda:
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(
                batch["ent_tok"],
                batch["opt_tok"],
                batch["ctx"],
                batch["ent_x"],
                batch["opt_x"],
                batch["glob_x"],
            )
            if is_cuda:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    emb = model.card_emb.weight.numel() + model.attack_emb.weight.numel()
    return total, total - emb  # (total, non-embedding ≈ compute-scaling)


def run(
    sizes: list[str], batches: list[int], regimes: list[tuple[int, int]], device: str, iters: int
) -> None:
    dev = torch.device(device)
    torch.set_grad_enabled(False)
    print(
        f"\n=== inference probe  device={device}  threads={torch.get_num_threads()}  "
        f"iters={iters} ===\n"
    )
    for sname in sizes:
        cfg = SIZE_BANDS[sname]
        model = PointerValueNet(cfg).to(dev).eval()
        total, nonemb = _params(model)
        print(
            f"[{cfg.name}] d={cfg.d_model} L={cfg.n_layers} h={cfg.n_heads} ff={cfg.d_ff}  "
            f"params={total / 1e6:.2f}M (non-emb {nonemb / 1e6:.2f}M)"
        )
        hdr = f"  {'batch':>5} {'n_ent':>5} {'n_opt':>5} {'ms/fwd':>8} {'ms/dec':>8} {'infer dec/s':>12}"
        print(hdr)
        for n_ent, n_opt in regimes:
            for b in batches:
                batch = _synthetic_batch(b, n_ent, n_opt, dev)
                sec = _time_forward(model, batch, iters)
                ms = sec * 1e3
                ms_dec = ms / b
                dec_s = b / sec
                print(f"  {b:>5} {n_ent:>5} {n_opt:>5} {ms:>8.2f} {ms_dec:>8.3f} {dec_s:>12,.0f}")
        # Combined throughput at the typical regime, biggest batch.
        # NB: this `device` figure is co-located inference (model runs on the SAME
        # resource as the env). On CPU that is the *submission* path and the
        # pessimistic floor. In training we DECOUPLE: CPU workers step envs while
        # the GPU batches inference, so the real training ceiling is
        # min(env_dec_s, gpu_infer_dec_s) — run this with --device cuda on Colab to
        # get gpu_infer_dec_s; the CPU row below is the worst case, not the ceiling.
        b = max(batches)
        sec = _time_forward(model, _synthetic_batch(b, regimes[0][0], regimes[0][1], dev), iters)
        infer_dec_s = b / sec
        print(
            f"  -> typical-regime infer throughput @batch{b} on {device}: {infer_dec_s:,.0f} dec/s"
        )
        for label, env in ENV_DEC_S.items():
            colocated = 1.0 / (1.0 / env + 1.0 / infer_dec_s)
            decoupled = min(env, infer_dec_s)
            print(
                f"     {label}: co-located≈{colocated:,.0f}/s | decoupled(GPU)≈min={decoupled:,.0f}/s"
            )
        print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sizes", default="tiny,small,medium", help=f"of {list(SIZE_BANDS)}")
    ap.add_argument("--batches", default="1,8,16,32", help="comma list of batch sizes")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    sizes = [s.strip() for s in a.sizes.split(",") if s.strip()]
    batches = [int(x) for x in a.batches.split(",") if x.strip()]
    # (n_entities, n_options): typical, busy-board, padded worst-case.
    regimes = [(16, 20), (24, 60), (30, 256)]
    run(sizes, batches, regimes, a.device, a.iters)
    print("Token-count regimes: (16,20) typical, (24,60) busy board, (30,256) padded worst case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
