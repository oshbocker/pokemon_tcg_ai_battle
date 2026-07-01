"""Phase 2 policy/value network: entity-token trunk + pointer actor + value head.

Consumes the arrays from `encoding.py` (one entity token per board object, one
candidate token per legal option) and implements the `docs/rl-obs-action.md` §5
contract:

  embeddings (card/attack/role/area/.../option-rank) → shared transformer trunk →
  **pointer actor** (scaled dot-product of a context query against candidate
  tokens — legality is free, no action mask) + **value head** (tanh ∈ [-1,1]).

Multi-pick decisions (`maxCount>1`) are handled by an **autoregressive pointer**
with a learned STOP key: pick a candidate, add it to the running query, re-score,
repeat until STOP (allowed once `minCount` met) or `maxCount`. The single-pick
case (the large majority of decisions) is just the first step.

`use_option_rank` (default on) gates the engine-order positional embedding — the
ablation lever for the B1-ordering prior (see `PHASE1_RESEARCH.md`). Flip it off to
train the counterfactual and measure whether the prior actually helps.

`use_card_meta` (default off) adds a frozen static card-metadata feature
(`card_meta.py`: type/HP/stage/ex-flags/... per Card ID) projected into the card
embeddings — the coevolution fix for novel-card deck mutations whose learned
embedding row is cold. The projection is zero-init and created last, so a meta-ON
net warm-started from a meta-OFF parent is behavior-identical at iteration 0.

This is the only canonical model definition; `scripts/bench_inference.py` imports
`SIZE_BANDS` / `PtcgNet` / `synthetic_collated` from here so timing reflects the
real network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .card_meta import CARD_META_DIM, build_card_meta_table
from .encoding import (
    ATTACK_VOCAB,
    CARD_VOCAB,
    ENTITY_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    N_AREA,
    N_ENERGY_TYPES,
    N_OPTION_RANK,
    N_OPTION_TYPE,
    N_ROLE,
    N_SELECT_CONTEXT,
    N_SPECIAL_COND,
    OPTION_FEAT_DIM,
    EncodedObs,
)

NEG_INF = -1e9  # masked-out logit


@dataclass
class ModelConfig:
    name: str = "small"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0
    use_option_rank: bool = True
    use_card_meta: bool = False


# Size bands probed in P0.3. `small` is the first training band (see PHASE0_THROUGHPUT.md).
SIZE_BANDS: dict[str, ModelConfig] = {
    "tiny": ModelConfig("tiny", 128, 4, 4, 512),
    "small": ModelConfig("small", 256, 6, 8, 1024),
    "medium": ModelConfig("medium", 384, 8, 8, 1536),
    "large": ModelConfig("large", 512, 10, 8, 2048),
}


# ---------------------------------------------------------------------------
# Batching: pad a list of EncodedObs into rectangular tensors + masks
# ---------------------------------------------------------------------------
def collate(batch: list[EncodedObs], device: torch.device | str = "cpu") -> dict:
    """Pad variable entity/option counts to the batch max and stack into tensors.

    Padding goes at the end, so original entity indices (used by `opt_target`) stay
    valid; padded option/entity slots are flagged in `*_mask` and never contribute.
    """
    b = len(batch)
    e_max = max(e.n_entities for e in batch)
    o_max = max(e.n_options for e in batch) if any(e.n_options for e in batch) else 1

    ent_role = np.zeros((b, e_max), np.int64)
    ent_card = np.zeros((b, e_max), np.int64)
    ent_feat = np.zeros((b, e_max, ENTITY_FEAT_DIM), np.float32)
    ent_energy = np.zeros((b, e_max, N_ENERGY_TYPES), np.float32)
    ent_mask = np.zeros((b, e_max), bool)

    opt_type = np.zeros((b, o_max), np.int64)
    opt_area = np.zeros((b, o_max), np.int64)
    opt_inplay = np.zeros((b, o_max), np.int64)
    opt_card = np.zeros((b, o_max), np.int64)
    opt_attack = np.zeros((b, o_max), np.int64)
    opt_special = np.zeros((b, o_max), np.int64)
    opt_rank = np.zeros((b, o_max), np.int64)
    opt_feat = np.zeros((b, o_max, OPTION_FEAT_DIM), np.float32)
    opt_target = np.zeros((b, o_max), np.int64)
    opt_mask = np.zeros((b, o_max), bool)

    glob = np.zeros((b, GLOBAL_FEAT_DIM), np.float32)
    ctx = np.zeros(b, np.int64)
    min_c = np.zeros(b, np.int64)
    max_c = np.zeros(b, np.int64)

    for i, e in enumerate(batch):
        ne, no = e.n_entities, e.n_options
        ent_role[i, :ne] = e.entity_role
        ent_card[i, :ne] = e.entity_card
        ent_feat[i, :ne] = e.entity_feat
        ent_energy[i, :ne] = e.entity_energy
        ent_mask[i, :ne] = True
        if no:
            opt_type[i, :no] = e.opt_type
            opt_area[i, :no] = e.opt_area
            opt_inplay[i, :no] = e.opt_inplay_area
            opt_card[i, :no] = e.opt_card
            opt_attack[i, :no] = e.opt_attack
            opt_special[i, :no] = e.opt_special
            opt_rank[i, :no] = e.opt_rank
            opt_feat[i, :no] = e.opt_feat
            opt_target[i, :no] = np.clip(e.opt_target, 0, ne - 1)
            opt_mask[i, :no] = True
        glob[i] = e.global_feat
        ctx[i], min_c[i], max_c[i] = e.context, e.min_count, e.max_count

    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return {
        "ent_role": t(ent_role),
        "ent_card": t(ent_card),
        "ent_feat": t(ent_feat),
        "ent_energy": t(ent_energy),
        "ent_mask": t(ent_mask),
        "opt_type": t(opt_type),
        "opt_area": t(opt_area),
        "opt_inplay": t(opt_inplay),
        "opt_card": t(opt_card),
        "opt_attack": t(opt_attack),
        "opt_special": t(opt_special),
        "opt_rank": t(opt_rank),
        "opt_feat": t(opt_feat),
        "opt_target": t(opt_target),
        "opt_mask": t(opt_mask),
        "glob": t(glob),
        "ctx": t(ctx),
        "min_c": t(min_c),
        "max_c": t(max_c),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PtcgNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Shared id embeddings (card embedding shared by entities and options).
        self.card_emb = nn.Embedding(CARD_VOCAB, d, padding_idx=0)
        self.attack_emb = nn.Embedding(ATTACK_VOCAB, d, padding_idx=0)
        self.role_emb = nn.Embedding(N_ROLE, d)
        self.area_emb = nn.Embedding(N_AREA, d)
        self.inplay_emb = nn.Embedding(N_AREA, d)
        self.opttype_emb = nn.Embedding(N_OPTION_TYPE, d)
        self.special_emb = nn.Embedding(N_SPECIAL_COND, d, padding_idx=0)
        self.ctx_emb = nn.Embedding(N_SELECT_CONTEXT, d)
        self.rank_emb = nn.Embedding(N_OPTION_RANK, d)
        # Numeric projections.
        self.ent_feat_proj = nn.Linear(ENTITY_FEAT_DIM, d)
        self.ent_energy_proj = nn.Linear(N_ENERGY_TYPES, d)
        self.opt_feat_proj = nn.Linear(OPTION_FEAT_DIM, d)
        self.global_proj = nn.Linear(GLOBAL_FEAT_DIM, d)
        # Trunk.
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        # Heads.
        self.query_proj = nn.Linear(d, d)
        self.selected_proj = nn.Linear(d, d)  # running "picked so far" for multi-pick
        self.stop_key = nn.Parameter(torch.zeros(d))  # learned STOP candidate
        # Pointer temperature: scale the q·k logits by 1/√d (docs/rl-obs-action.md §5).
        # Without it a d=256 dot product has std ~16, saturating the softmax at init →
        # near-zero entropy and an exploding first-update KL. Applied at EVERY logit
        # site (act/forward/evaluate_actions + the STOP key) so the act↔evaluate
        # distributions stay identical (the PPO-ratio parity invariant).
        self.logit_scale = d**-0.5
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1), nn.Tanh())
        # Frozen static card metadata (use_card_meta). Created LAST so a meta-ON net
        # draws the same RNG stream as a meta-OFF one for every shared module; with
        # the zero-init, bias-free projection the two are then behavior-identical at
        # init — warm-starting a meta-OFF parent into a meta-ON net (strict=False)
        # preserves the parent exactly at iter 0, and metadata influence is learned.
        # The table is a persistent buffer: checkpoints carry it, so inference (the
        # Kaggle bundle has no data/) never needs the CSV.
        if cfg.use_card_meta:
            try:
                table = torch.from_numpy(build_card_meta_table())
            except FileNotFoundError:  # no CSV here; a checkpoint load fills the buffer
                table = torch.zeros(CARD_VOCAB, CARD_META_DIM)
            self.card_meta_table: torch.Tensor  # (register_buffer is untyped)
            self.register_buffer("card_meta_table", table)
            self.card_meta_proj = nn.Linear(CARD_META_DIM, d, bias=False)
            nn.init.zeros_(self.card_meta_proj.weight)

    # -- shared encode: tokens -> trunk -> (option hidden, query base, value) --
    def _encode(self, batch: dict) -> dict:
        glob = self.global_proj(batch["glob"]).unsqueeze(1)  # [B,1,d] board context

        ent = (
            self.card_emb(batch["ent_card"])
            + self.role_emb(batch["ent_role"])
            + self.ent_feat_proj(batch["ent_feat"])
            + self.ent_energy_proj(batch["ent_energy"])
        )  # [B,E,d] -- pre-trunk entity embeddings (also the option->target source)
        if self.cfg.use_card_meta:
            ent = ent + self.card_meta_proj(self.card_meta_table[batch["ent_card"]])

        opt = (
            self.card_emb(batch["opt_card"])
            + self.attack_emb(batch["opt_attack"])
            + self.opttype_emb(batch["opt_type"])
            + self.area_emb(batch["opt_area"])
            + self.inplay_emb(batch["opt_inplay"])
            + self.special_emb(batch["opt_special"])
            + self.opt_feat_proj(batch["opt_feat"])
        )
        if self.cfg.use_card_meta:
            opt = opt + self.card_meta_proj(self.card_meta_table[batch["opt_card"]])
        # option -> target entity link (gather each option's target entity embedding)
        tgt = batch["opt_target"].unsqueeze(-1).expand(-1, -1, ent.shape[-1])  # [B,O,d]
        opt = opt + torch.gather(ent, 1, tgt)
        if self.cfg.use_option_rank:
            opt = opt + self.rank_emb(batch["opt_rank"])

        ent = ent + glob
        opt = opt + glob
        seq = torch.cat([ent, opt], dim=1)  # [B, E+O, d]
        pad = torch.cat([~batch["ent_mask"], ~batch["opt_mask"]], dim=1)  # True = pad
        h = self.trunk(seq, src_key_padding_mask=pad)

        n_ent = ent.shape[1]
        opt_h = h[:, n_ent:, :]  # [B,O,d]
        valid = (~pad).unsqueeze(-1).float()
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1.0)  # masked mean
        query = self.query_proj(pooled) + self.ctx_emb(batch["ctx"])  # [B,d]
        value = self.value_head(pooled).squeeze(-1)  # [B]
        return {"opt_h": opt_h, "opt_mask": batch["opt_mask"], "query": query, "value": value}

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step pointer logits [B,O] (masked) + value [B].

        This is the per-decision actor used directly for `maxCount==1` (the common
        case) and as the first step of the multi-pick loop."""
        enc = self._encode(batch)
        logits = torch.einsum("bod,bd->bo", enc["opt_h"], enc["query"]) * self.logit_scale
        logits = logits.masked_fill(~enc["opt_mask"], NEG_INF)
        return logits, enc["value"]

    # -- rollout: choose a (possibly multi-) selection per decision --
    @torch.no_grad()
    def act(
        self,
        encoded: list[EncodedObs],
        *,
        sample: bool = True,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> list[dict]:
        """Pick an action per decision. Returns one dict per input:
        {action: list[int], log_prob: float, value: float}. Honors min/max count via
        the autoregressive STOP pointer; `action` is always a valid engine selection.

        Single-pick decisions (`maxCount==1`, the overwhelming majority) are sampled
        in **one batched step** — critical for the distributed collector, where the
        old per-decision Python loop forced one GPU→CPU sync per decision and made
        rollout latency- rather than GPU-bound. The path mirrors
        `evaluate_actions`' onestep math, so act↔evaluate log-probs stay identical
        (the PPO-ratio parity invariant). Multi-pick decisions still replay the
        autoregressive loop per item."""
        batch = collate(encoded, device)
        enc = self._encode(batch)
        opt_h, query, value, opt_mask = enc["opt_h"], enc["query"], enc["value"], enc["opt_mask"]
        o_max = opt_h.shape[1]
        min_c, max_c = batch["min_c"], batch["max_c"]
        out: list[dict] = [{} for _ in encoded]

        # -- vectorised single-pick (maxCount==1): one batched sampling step --
        single = max_c == 1
        sidx = single.nonzero(as_tuple=True)[0]
        if sidx.numel():
            oh, q = opt_h[sidx], query[sidx]
            logits = (torch.einsum("bod,bd->bo", oh, q) * self.logit_scale).masked_fill(
                ~opt_mask[sidx], NEG_INF
            )
            stop = (q * self.stop_key).sum(-1, keepdim=True) * self.logit_scale
            stop = stop.masked_fill((min_c[sidx] != 0).unsqueeze(1), NEG_INF)  # stop only if min==0
            cat = torch.cat([logits, stop], dim=1)  # [ns, o_max+1]; col o_max = STOP
            lp = torch.log_softmax(cat, dim=-1)
            picks = (
                torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1)
                if sample
                else cat.argmax(dim=-1)
            )
            logp = lp.gather(1, picks[:, None]).squeeze(1)
            picks_l, logp_l = picks.tolist(), logp.tolist()
            val_l = value[sidx].tolist()
            for k, i in enumerate(sidx.tolist()):
                p = picks_l[k]
                action = [] if p == o_max else [p]  # o_max == STOP (only when min==0)
                out[i] = {"action": action, "log_prob": logp_l[k], "value": val_l[k]}

        # -- multi-pick (maxCount>1): autoregressive STOP pointer, per item --
        for i in (~single).nonzero(as_tuple=True)[0].tolist():
            e = encoded[i]
            n_opt = e.n_options
            keys = opt_h[i, :n_opt]  # [n_opt, d]
            q = query[i].clone()
            avail = torch.ones(n_opt, dtype=torch.bool, device=keys.device)
            chosen: list[int] = []
            logp_f = 0.0
            while len(chosen) < e.max_count and bool(avail.any()):
                allow_stop = len(chosen) >= e.min_count
                step = self._step_cat(keys, q, avail, allow_stop=allow_stop)
                probs = F.softmax(step, dim=-1)
                idx = (
                    int(torch.multinomial(probs, 1, generator=generator))
                    if sample
                    else int(torch.argmax(probs))
                )
                logp_f += float(torch.log(probs[idx].clamp_min(1e-12)))
                if allow_stop and idx == n_opt:  # STOP
                    break
                chosen.append(idx)
                avail[idx] = False
                q = q + self.selected_proj(keys[idx])
            out[i] = {"action": chosen, "log_prob": logp_f, "value": float(value[i])}
        return out

    # -- PPO: differentiable log-prob / entropy / value for taken actions --
    def evaluate_actions(
        self, batch: dict, actions: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score `actions` under the current params. Returns (log_prob[B],
        entropy[B], value[B]). Mirrors `act()`'s distribution exactly so that the
        ratio new/old is well-defined: a STOP candidate is appended whenever
        `minCount` is met. The common `maxCount==1` case is vectorised; the rare
        multi-pick case replays the autoregressive steps per sample."""
        enc = self._encode(batch)
        opt_h, query, value, opt_mask = enc["opt_h"], enc["query"], enc["value"], enc["opt_mask"]
        bsz, o_max, _ = opt_h.shape
        min_c, max_c = batch["min_c"], batch["max_c"]
        logp = query.new_zeros(bsz)
        entropy = query.new_zeros(bsz)

        onestep = max_c == 1  # covers single-pick and "pick up to one" (min_c==0)
        if bool(onestep.any()):
            idx = onestep.nonzero(as_tuple=True)[0]
            oh, q = opt_h[idx], query[idx]
            opt_logits = (torch.einsum("bod,bd->bo", oh, q) * self.logit_scale).masked_fill(
                ~opt_mask[idx], NEG_INF
            )
            stop = (q * self.stop_key).sum(-1, keepdim=True) * self.logit_scale  # [n,1]
            stop = stop.masked_fill((min_c[idx] != 0).unsqueeze(1), NEG_INF)  # stop only if min==0
            cat = torch.cat([opt_logits, stop], dim=1)  # [n, O+1]; col O = STOP
            lp_all = F.log_softmax(cat, dim=-1)
            tgt = torch.tensor(
                [(actions[j][0] if actions[j] else o_max) for j in idx.tolist()],
                device=cat.device,
            )
            logp[idx] = lp_all.gather(1, tgt[:, None]).squeeze(1)
            entropy[idx] = -(lp_all.exp() * lp_all).sum(1)

        for j in (~onestep).nonzero(as_tuple=True)[0].tolist():
            n = int(opt_mask[j].sum())
            lp, ent = self._replay_action(
                opt_h[j], query[j], n, int(min_c[j]), int(max_c[j]), actions[j]
            )
            logp[j], entropy[j] = lp, ent
        return logp, entropy, value

    def _replay_action(self, keys_all, q0, n, min_c, max_c, action):
        """Autoregressive log-prob + entropy of one multi-pick `action`."""
        keys = keys_all[:n]
        q = q0
        avail = torch.ones(n, dtype=torch.bool, device=keys.device)
        logp = q0.new_zeros(())
        entropy = q0.new_zeros(())
        for t, a in enumerate(action):
            cat = self._step_cat(keys, q, avail, allow_stop=(t >= min_c))
            lp = F.log_softmax(cat, dim=-1)
            logp = logp + lp[a]
            entropy = entropy + -(lp.exp() * lp).sum()
            avail[a] = False
            q = q + self.selected_proj(keys[a])
        if len(action) < max_c:  # a trailing STOP was chosen (allowed since len>=min_c)
            cat = self._step_cat(keys, q, avail, allow_stop=True)
            lp = F.log_softmax(cat, dim=-1)
            logp = logp + lp[-1]
            entropy = entropy + -(lp.exp() * lp).sum()
        return logp, entropy

    def _step_cat(self, keys, q, avail, *, allow_stop: bool):
        logits = (keys @ q * self.logit_scale).masked_fill(~avail, NEG_INF)
        if allow_stop:
            stop = (self.stop_key * q).sum() * self.logit_scale
            return torch.cat([logits, stop.view(1)])
        return logits


# ---------------------------------------------------------------------------
# Synthetic batch (for the inference probe; bypasses the engine)
# ---------------------------------------------------------------------------
def synthetic_collated(
    b: int,
    n_ent: int,
    n_opt: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict:
    """Random padded batch with the right shapes/dtypes for forward-pass timing."""
    g = torch.Generator().manual_seed(seed)

    def ri(hi, *shape):
        return torch.randint(0, hi, shape, generator=g)

    return {
        "ent_role": ri(N_ROLE, b, n_ent).to(device),
        "ent_card": ri(CARD_VOCAB, b, n_ent).to(device),
        "ent_feat": torch.rand(b, n_ent, ENTITY_FEAT_DIM, generator=g).to(device),
        "ent_energy": torch.rand(b, n_ent, N_ENERGY_TYPES, generator=g).to(device),
        "ent_mask": torch.ones(b, n_ent, dtype=torch.bool, device=device),
        "opt_type": ri(N_OPTION_TYPE, b, n_opt).to(device),
        "opt_area": ri(N_AREA, b, n_opt).to(device),
        "opt_inplay": ri(N_AREA, b, n_opt).to(device),
        "opt_card": ri(CARD_VOCAB, b, n_opt).to(device),
        "opt_attack": ri(ATTACK_VOCAB, b, n_opt).to(device),
        "opt_special": ri(N_SPECIAL_COND, b, n_opt).to(device),
        "opt_rank": torch.minimum(torch.arange(n_opt), torch.tensor(N_OPTION_RANK - 1))
        .expand(b, n_opt)
        .contiguous()
        .to(device),
        "opt_feat": torch.rand(b, n_opt, OPTION_FEAT_DIM, generator=g).to(device),
        "opt_target": ri(n_ent, b, n_opt).to(device),
        "opt_mask": torch.ones(b, n_opt, dtype=torch.bool, device=device),
        "glob": torch.rand(b, GLOBAL_FEAT_DIM, generator=g).to(device),
        "ctx": ri(N_SELECT_CONTEXT, b).to(device),
        "min_c": torch.ones(b, dtype=torch.int64, device=device),
        "max_c": torch.ones(b, dtype=torch.int64, device=device),
    }


def param_counts(model: PtcgNet) -> tuple[int, int]:
    """(total, non-embedding) parameter counts; non-emb ≈ the compute-scaling part."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.card_emb.weight.numel() + model.attack_emb.weight.numel()
    return total, total - emb
