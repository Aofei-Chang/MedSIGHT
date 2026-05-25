#!/usr/bin/env bash
# Score MedSIGHT predictions on SLAKE.
#
# Usage: scripts/eval_slake.sh <run_name> [<gt_path>] [<pred_dir>]
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name>}"
GT="${2:-/path/to/SLAKE/test.json}"
PRED_DIR="${3:-./outputs/eval/SLAKE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

python run_eval.py \
    --gt "$GT" \
    --pred "$PRED_DIR/answers_${RUN_NAME}.jsonl" \
    --eval_res "$PRED_DIR/eval_${RUN_NAME}.txt"
