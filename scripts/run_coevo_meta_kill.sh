#!/usr/bin/env bash
# Novel-card kill-criterion (coevolution §4.1 follow-up): does the frozen
# card-metadata feature (use_card_meta) rescue warm-start fine-tuning on a deck
# mutation that adds a card the parent NEVER saw (cold embedding row)?
#
# Deck: agent/decks/archaludon_genesect_swap.csv — parent Archaludon deck with
# 1x Relicanth (57) -> 1x Genesect ex (547; novel across ALL training decks).
# Parent: outputs/probe_archaludon_medium_long/best.pt (trained meta-OFF; the
# ON arm grafts a zero-init metadata pathway, behavior-identical at iter 0).
# 2x2: {warm-OFF, warm-ON} fine-tunes (same recipe as the judge_swap kill run:
# 30 iters, 48 g/iter, LR 3e-5->1e-5, mixed_pool league, pool gate >55%), then
# high-n side-swapped evals: zero-shot / warm-OFF / warm-ON (ref = the parent-on-
# original eval already in outputs/coevo_kill/eval_ref.csv).
# Usage: bash scripts/run_coevo_meta_kill.sh [GAMES]
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-160}"
PARENT_CKPT="outputs/probe_archaludon_medium_long/best.pt"
SWAP_DECK="agent/decks/archaludon_genesect_swap.csv"
OPPS="kaggle:archaludon,kaggle:starmie,kaggle:dragapult,kaggle:alakazam,kaggle:romanrozen_v10,heuristic,random"
OUT=outputs/coevo_meta
mkdir -p "$OUT"

train() { # subdir extra-flags...
  local sub="$1"; shift
  echo "===== TRAIN $sub  $(date) ====="
  uv run python scripts/train_selfplay.py \
    --deck "$SWAP_DECK" \
    --opponent self --league --collector dist --workers 7 \
    --iters 30 --games-per-iter 48 \
    --init-ckpt "$PARENT_CKPT" \
    --lr 3e-5 --lr-final 1e-5 \
    --gate --gate-vs pool --gate-every 10 --gate-games 30 --gate-threshold 0.55 \
    --eval-every 10 --eval-games 80 \
    --out "$OUT/$sub" --device cpu --seed 0 "$@"
}

# best.pt exists only if the gate promoted; fall back to last.pt (report honestly).
ckpt_of() { if [ -f "$OUT/$1/best.pt" ]; then echo "$OUT/$1/best.pt"; else echo "$OUT/$1/last.pt"; fi; }

run_eval() { # label champion out_csv
  echo "===== EVAL $1  $(date) ====="
  uv run python scripts/eval.py --champion "$2" --deck "$SWAP_DECK" \
    --opponents "$OPPS" --games "$GAMES" --out "$3"
}

train warm_off              2>&1 | tee rl_research/coevo_meta_warm_off.log
train warm_on --use-card-meta 2>&1 | tee rl_research/coevo_meta_warm_on.log

{
  run_eval zeroshot-parent-on-swap "model:$PARENT_CKPT"        "$OUT/eval_zeroshot.csv"
  run_eval warm-off-on-swap        "model:$(ckpt_of warm_off)" "$OUT/eval_warm_off.csv"
  run_eval warm-on-on-swap         "model:$(ckpt_of warm_on)"  "$OUT/eval_warm_on.csv"
  echo "ALL DONE -> $OUT/eval_{zeroshot,warm_off,warm_on}.csv  (ref: outputs/coevo_kill/eval_ref.csv)"
} 2>&1 | tee rl_research/coevo_meta_eval.log
