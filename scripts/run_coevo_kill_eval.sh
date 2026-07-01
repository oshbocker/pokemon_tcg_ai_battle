#!/usr/bin/env bash
# §4.1 kill-criterion evals: parent-on-original (ref), parent-on-swapped (zero-shot),
# warm-best, cold-best — each vs the 7 mixed_pool opponents, side-swapped, Wilson CIs.
# Usage: bash scripts/run_coevo_kill_eval.sh [GAMES]
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-200}"
PARENT="model:outputs/probe_archaludon_medium_long/best.pt"
ORIG_DECK="agent/kaggle_agents/archaludon_deck.csv"
SWAP_DECK="agent/decks/archaludon_judge_swap.csv"
OPPS="kaggle:archaludon,kaggle:starmie,kaggle:dragapult,kaggle:alakazam,kaggle:romanrozen_v10,heuristic,random"
OUT=outputs/coevo_kill
mkdir -p "$OUT"

run() { # label champion deck out_csv
  echo "===== EVAL $1 ====="
  uv run python scripts/eval.py --champion "$2" --deck "$3" \
    --opponents "$OPPS" --games "$GAMES" --out "$4"
}

run ref-parent-on-original "$PARENT"                              "$ORIG_DECK" "$OUT/eval_ref.csv"
run zeroshot-parent-on-swap "$PARENT"                             "$SWAP_DECK" "$OUT/eval_zeroshot.csv"
run warm-best-on-swap      "model:$OUT/warm/best.pt"              "$SWAP_DECK" "$OUT/eval_warm.csv"
run cold-best-on-swap      "model:$OUT/cold/best.pt"              "$SWAP_DECK" "$OUT/eval_cold.csv"
echo "ALL EVALS DONE -> $OUT/eval_{ref,zeroshot,warm,cold}.csv"
