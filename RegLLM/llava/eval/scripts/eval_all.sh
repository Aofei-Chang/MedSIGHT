#!/usr/bin/env bash
# Convenience wrapper: run inference + scoring for every dataset.
#
# Usage: scripts/eval_all.sh <run_name>
set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_inference.sh" all "$RUN_NAME"
bash "$SCRIPT_DIR/eval_vqa_rad.sh"     "$RUN_NAME"
bash "$SCRIPT_DIR/eval_slake.sh"       "$RUN_NAME"
bash "$SCRIPT_DIR/eval_pathvqa.sh"     "$RUN_NAME"
bash "$SCRIPT_DIR/eval_biomedparse.sh" "$RUN_NAME"
bash "$SCRIPT_DIR/eval_omnimed.sh"     "$RUN_NAME"
