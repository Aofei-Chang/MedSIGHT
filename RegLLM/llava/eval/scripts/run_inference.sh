#!/usr/bin/env bash
# Run MedSIGHT inference on one (or all) datasets defined in configs/datasets.yaml.
#
# Usage:
#   scripts/run_inference.sh <dataset> <run_name>
#
# Examples:
#   scripts/run_inference.sh VQA-RAD     baseline
#   scripts/run_inference.sh OmniMedVQA  baseline
#   scripts/run_inference.sh all         baseline
set -euo pipefail

DATASET="${1:?Usage: $0 <dataset> <run_name>}"
RUN_NAME="${2:?Usage: $0 <dataset> <run_name>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_CONFIG="${MODEL_CONFIG:-$EVAL_DIR/configs/model.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-$EVAL_DIR/configs/datasets.yaml}"

cd "$EVAL_DIR"

python -m llava.eval.inference \
    --model-config   "$MODEL_CONFIG" \
    --dataset-config "$DATASET_CONFIG" \
    --dataset        "$DATASET" \
    --run-name       "$RUN_NAME"
