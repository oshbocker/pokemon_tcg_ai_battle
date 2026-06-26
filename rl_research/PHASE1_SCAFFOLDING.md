# Phase 1 — Scaffolding & contracts (2026-06-26)

Dated experiment-log entry (Lesson 10: scaffold the agentic loop before
generating lots of code). Phase 1 set up the guardrails + encoding contract the
Phase-2 model is built against, and — because it gated the last Phase-0 exit
criterion — ran the **P0.3 inference probe** and stood up the **Phase-5 eval
skeleton** early. Companion findings live in
[`PHASE0_THROUGHPUT.md`](./PHASE0_THROUGHPUT.md); plan in
[`SELFPLAY_RL_PLAN.md`](./SELFPLAY_RL_PLAN.md).

## What shipped

| Area | Artifact | Notes |
|---|---|---|
| Encoder | `src/ptcg_battle/encoding.py` | raw obs dict → entity + option tokens; torch-free (numpy) |
| Contract | `docs/rl-obs-action.md` | the spec the model implements against (§5 = model interface) |
| Tests | `tests/test_encoding.py` (9) + `tests/conftest.py` | round-trip invariants; engine-driven fixtures |
| Eval | `src/ptcg_battle/eval_harness.py`, `scripts/eval.py` | high-n side-swapped unpaired; Wilson CIs; resumable CSV |
| Inference probe | `scripts/bench_inference.py` | P0.3; pointer+value transformer at 3 size bands |
| Gate | `scripts/prepare.py`, `justfile` | format→lint→pyright→pytest, fail-fast |
| Deps | `pyproject.toml` | `pytest` in `dev`; `torch` in optional `rl` extra (CPU/CUDA note) |

## Key decisions & their *why*

- **Encode from the raw JSON dict, never the dataclass.** Phase-0 finding #2:
  `to_observation_class()` ~halves throughput. The encoder indexes `obs[...]`
  directly and is **torch-free** so encoding tests + CPU rollout workers don't
  pull the heavy `rl` extra. Card knowledge lives in a learned **card-ID
  embedding** (1267 cards) — not hand-coded features (Lesson 3).
- **Pointer head, candidate-token-per-option, 1:1.** The engine only offers legal
  options, so legality is free — no action mask (Lesson 4). The headline tested
  invariant: *every legal option maps to exactly one candidate token*, index-
  aligned with `select.option`. Each candidate also carries an `opt_target` link
  to the entity token it acts on (e.g. ATTACH energy → receiving Pokémon).
- **First model band = `small` (d256/L6, ~5.6M).** P0.3 CPU floor: ~3.4 ms/decision
  typical, ~25 ms worst-case (256 options) at batch 1 — CPU-submission-safe today
  and matches the plan's "start ~1–5M to validate the pipeline." `tiny` (1.2M) is
  the time-budget fallback (P6.3); `medium`/`large` are Phase-4 scaling targets
  (GPU inference + quantise for the bundle).
- **Eval = volume + side-swap, never paired seeds** (engine unseedable, P0.4).
  Measured **A/A null = 49.8% over 998 games, ±3.1pp** — *on* 50%, so side-swap
  alone cancelled the seat bias (it did **not** come in wider than Orbit Wars at
  equal n). Don't-act floor: **~1.5K games/arm for a 5 pp edge**, ~4.3K for 3 pp.
  Sanity: heuristic beats `random` 95.7% and `first` 94.9%.

## Verification

`uv run python scripts/prepare.py` → all green (ruff format, ruff check, pyright
basic, 9 pytest). The encoder was smoke-tested across all 11 SelectContexts seen
in real mirror games (incl. multi-pick `maxCount>1` and the 34-option max), and
`bench_inference.py` / `eval.py` both run end-to-end.

## Open follow-ups (carry into Phase 2)

- **GPU inference number.** Run `bench_inference.py --device cuda` on a Colab L4
  and Blackwell to fill in `gpu_infer_dec_s` and confirm env stays the training
  bottleneck (the decoupled ceiling is `min(env_dec_s, gpu_infer_dec_s)`).
- **Per-turn time limit (TBD).** Re-pull competition pages (P6.1) to turn the
  P0.3 per-decision latencies into a hard go/no-go for `medium`+ on CPU.
- **Multi-pick head.** Spec'd as autoregressive pointer in `rl-obs-action.md` §5;
  validate when the policy is built (P2.3).
- **Opponent suite.** Add the strong public kernels to eval once a trained
  `model:<path>` agent exists (P5.1).

## Next (Phase 2)

Build the `small` model: embeddings + shared transformer trunk + pointer actor +
value head (the `docs/rl-obs-action.md` §5 shape, already prototyped for timing in
`bench_inference.py`), then single-process rollout + a PPO update that learns to
beat `random` decisively — the end-to-end sanity check before any scale.
