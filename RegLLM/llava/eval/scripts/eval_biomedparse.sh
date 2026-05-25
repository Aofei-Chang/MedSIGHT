#!/usr/bin/env bash
# Score MedSIGHT predictions on BiomedParse (SegVQA), with per-modality breakdown.
#
# Usage: scripts/eval_biomedparse.sh <run_name> [<gt_path>] [<pred_dir>]
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name>}"
GT="${2:-/path/to/BiomedParse/SegVQA_Diagnostic_test_vqa.json}"
PRED_DIR="${3:-./outputs/eval/BiomedParse}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

python run_eval.py \
    --gt "$GT" \
    --pred "$PRED_DIR/answers_${RUN_NAME}.jsonl" \
    --eval_res "$PRED_DIR/eval_${RUN_NAME}.txt" \
    --modality_split
