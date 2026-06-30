#!/usr/bin/env python
"""Package a trained RL checkpoint into a competition-ready `submission.tar.gz`.

Unlike `build_submission.py` (heuristic agents = main.py + deck.csv + cg/), an RL agent
also needs its **torch checkpoint + the model/encoder code** in the bundle. This vendors
`src/ptcg_battle/{encoding,model}.py` into an `rl/` package alongside the checkpoint, so
the submission is self-contained (torch itself is provided by the kaggle-environments
runtime image). Archive layout, all at the root:

    main.py        # the RL entry point (agent/rl_main.py)
    deck.csv       # the deck the policy pilots
    cg/            # the cabt engine SDK
    rl/__init__.py
    rl/encoding.py # vendored, numpy-only
    rl/model.py    # vendored, torch (imports .encoding)
    rl/best.pt     # the {model, cfg} checkpoint

    uv run python scripts/build_rl_submission.py \
        --checkpoint outputs/probe_archaludon_medium_long/best.pt \
        --deck agent/kaggle_agents/archaludon_deck.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "ptcg_battle"

# Post-build gate: run the bundle EXACTLY as Kaggle does — exec main.py with NO `__file__`
# (Kaggle `exec()`s the agent into a bare globals dict; a module-level `__file__` ref there
# is a NameError that crashes before agent() ever runs — this is the bug that errored our
# first RL submission). cwd = the unpacked bundle, only the bundle on the path. Then play a
# full self-game and assert every selection is legal. Hard-fails on import crash / illegal
# move; if torch is in the build env it also requires the RL policy to actually load.
_VALIDATE_SRC = r"""
import os, sys
bundle = sys.argv[1]
os.chdir(bundle)
sys.path.insert(0, bundle)
g = {"__name__": "__main__"}  # fresh globals, NO __file__ — exactly like Kaggle's exec(code, env)
exec(compile(open("main.py").read(), "main.py", "exec"), g)
agent = g["agent"]
model_loaded = g.get("_MODEL") is not None
deck = agent({"select": None})
assert isinstance(deck, list) and len(deck) == 60, f"initial selection must be 60 cards, got {deck!r:.60}"
from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish
obs, _ = battle_start(deck, deck); illegal = decisions = 0; cur = None
try:
    for _ in range(4000):
        oc = to_observation_class(obs); cur = oc.current
        if cur is None or (cur.result is not None and cur.result >= 0): break
        sel = obs["select"]; n = len(sel.get("option") or [])
        act = agent(obs); decisions += 1
        if not (all(isinstance(i, int) and 0 <= i < n for i in act) and len(set(act)) == len(act)
                and sel["minCount"] <= len(act) <= sel["maxCount"]):
            illegal += 1
finally:
    try: battle_finish()
    except Exception: pass
assert decisions > 0, "agent took no decisions"
assert illegal == 0, f"{illegal}/{decisions} ILLEGAL selections"
try:
    import torch; has_torch = True
except Exception:
    has_torch = False
if has_torch and not model_loaded:
    raise SystemExit("torch is present but the RL policy did not load (agent fell back) — bundle is broken")
print(f"VALIDATE_OK decisions={decisions} model_loaded={model_loaded} torch_in_build_env={has_torch}")
"""


def validate_bundle(tar: Path) -> None:
    """Unpack the built tar and run it like the Kaggle runtime; raise if it would error."""
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(tar, "r:gz") as t:
            t.extractall(td)  # noqa: S202 — our own freshly-built archive
        proc = subprocess.run(
            [sys.executable, "-c", _VALIDATE_SRC, td],
            capture_output=True,
            text=True,
            timeout=300,
        )
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 or "VALIDATE_OK" not in proc.stdout:
        raise RuntimeError(f"bundle validation FAILED (would error on Kaggle):\n{out}")
    line = next(ln for ln in proc.stdout.splitlines() if "VALIDATE_OK" in ln)
    print(f"validated: {line}")
    if "model_loaded=False" in line:
        print(
            "  WARNING: RL policy did NOT load (torch absent in build env) — only the "
            "no-crash + legality gate ran. Re-run with the `rl` extra to exercise the model."
        )


def build(main: Path, deck: Path, cg: Path, checkpoint: Path, out: Path) -> Path:
    for label, p in [("main", main), ("deck", deck), ("cg", cg), ("checkpoint", checkpoint)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found at {p}")
    deck_ids = [ln for ln in deck.read_text().splitlines() if ln.strip()]
    if len(deck_ids) != 60:
        raise ValueError(f"deck must have exactly 60 Card IDs, got {len(deck_ids)}")

    def _clean(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if "__pycache__" in Path(info.name).parts or info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        (stage / "rl").mkdir()
        (stage / "rl" / "__init__.py").write_text("")
        shutil.copy2(SRC / "encoding.py", stage / "rl" / "encoding.py")
        shutil.copy2(SRC / "model.py", stage / "rl" / "model.py")
        shutil.copy2(checkpoint, stage / "rl" / "best.pt")

        with tarfile.open(out, "w:gz") as tar:
            tar.add(main, arcname="main.py")
            tar.add(deck, arcname="deck.csv")
            tar.add(cg, arcname="cg", filter=_clean)
            tar.add(stage / "rl", arcname="rl", filter=_clean)

    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    size_mb = out.stat().st_size / 1e6
    print(f"created {out}  ({size_mb:.1f} MB)")
    for n in sorted(names):
        print("  ", n)
    if "main.py" not in names:
        raise RuntimeError("main.py must be at the archive root")
    for need in ("rl/model.py", "rl/encoding.py", "rl/best.pt", "deck.csv"):
        if need not in names:
            raise RuntimeError(f"missing {need} in bundle")
    return out


def build_and_validate(
    main: Path, deck: Path, cg: Path, checkpoint: Path, out: Path, validate: bool = True
) -> Path:
    build(main, deck, cg, checkpoint, out)
    if validate:
        validate_bundle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--main", default=REPO / "agent" / "rl_main.py", type=Path)
    ap.add_argument("--deck", required=True, type=Path, help="60-card deck the policy pilots")
    ap.add_argument("--cg", default=REPO / "agent" / "cg", type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path, help="{model,cfg} .pt checkpoint")
    ap.add_argument("--out", default=REPO / "submissions" / "submission_rl.tar.gz", type=Path)
    ap.add_argument(
        "--no-validate", action="store_true", help="skip the Kaggle-faithful post-build gate"
    )
    a = ap.parse_args()
    build_and_validate(a.main, a.deck, a.cg, a.checkpoint, a.out, validate=not a.no_validate)


if __name__ == "__main__":
    main()
