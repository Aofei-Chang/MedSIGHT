#!/bin/bash
# Stage-1 alignment training for the Qwen2.5-7B MedSIGHT variant.
#
# Same multimodal projector pretraining as `align_regtok_1110.sh` but with
# Qwen2.5-7B as the base LLM instead of Qwen3-8B. Used for the rebuttal
# experiments that compare backbone choice.
#
# Required paths (override via environment variables before running):
#   REGTOK_ROOT     – path to this repository
#   DATA_ROOT       – root directory containing the alignment data
#   CKPT_ROOT       – directory to write checkpoints into
#   REGTOK_WEIGHTS  – pretrained Region Tokenizer weights
#   VISION_TOWER    – pretrained UniMed-CLIP weights
#   HF_HUB_CACHE    – HuggingFace cache

set -euo pipefail

export FAST_TRANSFORMER=O0
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGTOK_ROOT="${REGTOK_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
CKPT_ROOT="${CKPT_ROOT:-./checkpoints}"
REGTOK_WEIGHTS="${REGTOK_WEIGHTS:-/path/to/RegTok/checkpoints/regtok_weights.pt}"
VISION_TOWER="${VISION_TOWER:-/path/to/CLIPs/unimed_clip_vit_l14.pt}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface}"

PRETRAIN_OUT_PATH="${CKPT_ROOT}/i2t_pre_region_qwen2"
PRETRAIN_TASK_NAME="$(basename "${PRETRAIN_OUT_PATH%/}")"

WORKER_NUM="${WORKER_NUM:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

torchrun \
    --nnodes "$WORKER_NUM" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr localhost \
    --master_port 16668 \
    llava/train/train_mem.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --version qwen \
    --data_path "${DATA_ROOT}/LVLM/HuatuoGPT/Filter_PubMedVision_Alignment_VQA.json" \
    --image_folder "${DATA_ROOT}/LVLM/HuatuoGPT" \
    --vision_tower "$VISION_TOWER" \
    --mm_vision_vq_type RegTok \
    --regtok_config_path "${REGTOK_ROOT}/source/regtok/regtok_config.yaml" \
    --regtok_weight_path "$REGTOK_WEIGHTS" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_tuning_embedding False \
    `# Selects the UniMed-CLIP block fed into the region perceiver. Must match the` \
    `# value in configs/model.yaml at inference time.` \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "$PRETRAIN_OUT_PATH" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "$PRETRAIN_TASK_NAME" \
    --use_region_tokens True \
    --use_sep_proj False
