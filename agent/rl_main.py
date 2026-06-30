"""Kaggle submission entry point for a trained self-play RL checkpoint.

Bundle layout (all at the archive root):
    main.py        # this file
    deck.csv       # the 60-card deck the policy was trained to pilot
    cg/            # the cabt engine SDK
    rl/            # vendored: encoding.py + model.py + best.pt (torch checkpoint)

Contract: `agent(obs_dict) -> list[int]`. At initial selection (`obs.select is None`)
return the 60 Card IDs; otherwise return option indices. The policy is greedy
(`sample=False`). Everything is wrapped so the agent **never crashes**: if torch or the
checkpoint is unavailable in the eval runtime, or inference raises, it falls back to a
legal in-bounds selection read straight from the raw obs dict. The encoder reads the raw
JSON dict (never the dataclass) — same hot-loop path used in training/eval.
"""

import contextlib
import os
import sys

# Kaggle exec()s this file (no `__file__`), so resolve our dir defensively: the runtime
# unpacks the bundle at /kaggle_simulations/agent; fall back to cwd for local runs.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = "/kaggle_simulations/agent"
if not os.path.isdir(_HERE):
    _HERE = os.getcwd()
for _p in (_HERE, "/kaggle_simulations/agent", os.getcwd()):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def _read_deck():
    for fp in (
        os.path.join(_HERE, "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
        "deck.csv",
    ):
        if os.path.exists(fp):
            with open(fp) as f:
                ids = [int(x) for x in f.read().split() if x.strip()]
            if len(ids) == 60:
                return ids
    raise FileNotFoundError("deck.csv (a legal 60-card deck)")


MY_DECK = _read_deck()

# --- Load the RL policy once; degrade gracefully if anything is missing. ---
_MODEL = None
_TORCH = None
_ENCODE = None
try:
    import torch

    with contextlib.suppress(Exception):
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    from rl.encoding import encode_observation as _ENCODE
    from rl.model import ModelConfig, PtcgNet

    _ckpt = torch.load(os.path.join(_HERE, "rl", "best.pt"), map_location="cpu", weights_only=False)
    _cfg = ModelConfig(**_ckpt["cfg"]) if isinstance(_ckpt.get("cfg"), dict) else ModelConfig()
    _MODEL = PtcgNet(_cfg)
    _MODEL.load_state_dict(_ckpt["model"])
    _MODEL.eval()
    _TORCH = torch
except Exception:
    _MODEL = None  # fall back to the legal policy below


def _bounds(sel, n):
    lo = sel.get("minCount", 1) or 0
    hi = sel.get("maxCount", 1) or 1
    lo = max(0, min(lo, n))
    hi = max(lo, min(hi, n))
    return lo, hi


def _legal_fallback(obs_dict):
    """Never-crash legal selection straight from the raw dict (first-k, clamped)."""
    sel = obs_dict.get("select")
    if sel is None:
        return MY_DECK
    n = len(sel.get("option") or [])
    if n == 0:
        return []
    lo, hi = _bounds(sel, n)
    return list(range(min(max(1, lo), n)))


def _clamp_action(action, sel):
    """Dedup, drop out-of-bounds, top up to minCount, cap at maxCount."""
    n = len(sel.get("option") or [])
    lo, hi = _bounds(sel, n)
    seen, out = set(), []
    for i in action or []:
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    if len(out) < lo:
        for i in range(n):
            if i not in seen:
                out.append(i)
                if len(out) >= lo:
                    break
    return out[:hi] if hi else out


def agent(obs_dict):
    if obs_dict.get("select") is None:
        return MY_DECK
    if _MODEL is not None:
        try:
            with _TORCH.no_grad():
                out = _MODEL.act([_ENCODE(obs_dict)], sample=False, device="cpu")
            return _clamp_action(out[0]["action"], obs_dict["select"])
        except Exception:
            return _legal_fallback(obs_dict)
    return _legal_fallback(obs_dict)
