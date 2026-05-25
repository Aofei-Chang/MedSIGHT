#!/usr/bin/env bash
# Score MedSIGHT predictions on OmniMedVQA (multiple-choice, per modality).
#
# Usage: scripts/eval_omnimed.sh <run_name> [<qa_root>] [<pred_dir>]
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name>}"
QA_ROOT="${2:-/path/to/OmniMedVQA/QA_information/Open-access}"
PRED_DIR="${3:-./outputs/eval/OmniMedVQA}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

MODALITIES=(CT OCT X-Ray MRI ultrasound Microscopy Fundus)

for MOD in "${MODALITIES[@]}"; do
    python run_eval_mc.py \
        --gt "$QA_ROOT/Modality_${MOD}.json" \
        --pred "$PRED_DIR/answers_${MOD}_${RUN_NAME}.jsonl" \
        --eval_res "$PRED_DIR/eval_${MOD}_${RUN_NAME}.txt"
done
