# Archaludon self-play go/no-go probe — first L4 results + the eval bug

Date: **2026-06-30**. The deck-controlled, held-out-yardstick probe from
[`INTERMEDIATE_SELFPLAY_2026-06-29.md`](./INTERMEDIATE_SELFPLAY_2026-06-29.md), run on the
L4. Question: can self-play **beat a strong, deck-matched rule agent** (the load-bearing
thesis)? Trainee = Archaludon on the rule agent's exact 60-card list
(`agent/kaggle_agents/archaludon_deck.csv`); `kaggle:archaludon` held OUT of the training
pool (`mixed_pool_heldout_archaludon.json`) as the only yardstick.

## Three findings

**1. Stability: SOLVED (orbit-wars recipe).** The first `medium` (15.8M) run collapsed —
entropy → 0 by it8, KL thrashing, `upd=2` — because LR 3e-4 + tiny batch drove the bigger
net deterministic before the ratcheting entropy controller could react (its docstring
predicts exactly this). The fix from the Orbit Wars write-ups
([[orbit-wars-selfplay-lessons]]) — **LR 1e-4, games/iter 256, epochs 1, ent-coef floor
0.04, target-entropy 0.15, gate 0.65** (grad-clip 0.5 + adv-norm already on) — trained
`medium_v2` cleanly to it200: entropy 0.06 → 0.61, KL < 0.1 late, full `upd≈70`.

**2. The in-loop eval was lying (now fixed).** `quick_eval` → `_make_fixed_opponent` only
handled `heuristic`/`random`/`first`; **every other spec silently fell through to the
'first' baseline**, on the trainee deck. So both probes' `mirror` and `kaggle:archaludon`
columns (the "92–95%") were the trivial 'first' opponent — the hand-coded Archaludon was
*never tested in-loop*. Tell: those two columns track each other across all 200 iters
(same opponent; differ only by unseedable engine RNG). **Fix (committed):** `quick_eval`
now routes `kaggle:<name>`/`heuristic` through `_resolve_eval_opponent` (own deck, proper
side-swap) and **raises on unknown specs** (incl. `mirror`) instead of degrading silently.
Verified: `medium_v2` best.pt now reports `kaggle:archaludon` ~30–37% in-loop (matches
eval.py), not 92%.

**3. The real number — and capacity is NOT the lever.** True held-out eval via
`scripts/eval.py` (n=160, side-swapped, Wilson CI), best.pt:

| ckpt | vs `kaggle:archaludon` | vs `heuristic` | vs `random` |
|------|----------------------:|---------------:|------------:|
| `small` (5.7M)   | **25.6%** [19.5, 32.9] | 66.2% | 99.4% |
| `medium_v2` (15.8M) | **28.7%** [22.3, 36.2] | 63.7% | 100% |

Both **lose** to the strong matched rule agent, and 3× the params barely moved it
(25.6 → 28.7, CIs overlap). **The bottleneck is steps, not capacity.** We trained ~3M
decisions (200 iters × 256 games × ~60 dec); Gerar hit top-2% at **412M steps with a 750K
net** (~130×), the big runs at 10–15B. Verdict: **not an algorithmic NO-GO — badly
undertrained.** Stability is solved, so we can scale.

## Next: the scaled long-run (the actual go/no-go)

`notebooks/colab_selfplay.ipynb` (Phase 4b) now runs `medium` on the stable recipe at
**~6× the steps** (`ITERS=1200`, ~12–16h at ~40s/iter) with the **honest** in-loop eval
(`--eval-opponents random,heuristic,kaggle:archaludon`; `mirror` dropped — it now raises).
Read the real `kaggle:archaludon` slope every 25 iters: climbing through ~40–50% and rising
→ **GO**; stuck in the 25–35% band → steps alone won't close it → **BC warm-start**
([[bc-pool-plan]]) is the next lever. No `--resume` yet → treat as one long session.

Banked: `outputs/probe_archaludon_{small,medium_v2}/{best,last}.pt` (frozen pool assets);
logs `rl_research/colab_probe_archaludon_{small,medium_v2}.txt`; eval CSVs
`outputs/eval/probe_{small,medium_v2}_real.csv`.
