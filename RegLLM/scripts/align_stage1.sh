#!/bin/bash
# Stage-1 alignment pretraining for MedSIGHT (Qwen3-8B + RegTok).
#
# Pretrains the multimodal projector on the PubMedVision alignment corpus.
# The projector produced here is the input to the stage-2 segmentation
# alignment in `align_regtok_1110.sh`.
#
# Required paths (override via environment variables before running):
#   REGTOK_ROOT     – path to this repository (defaults to current working dir)
#   DATA_ROOT       – root directory containing the alignment data
#   CKPT_ROOT       – directory to write checkpoints into
#   REGTOK_WEIGHTS  – pretrained Region Tokenizer weights
#   VISION_TOWER    – pretrained UniMed-CLIP weights
#   HF_HUB_CACHE    – HuggingFace cache (Qwen3 weights live here)
#   WANDB_PROJECT   – Weights & Biases project name
#   WANDB_ENTITY    – Weights & Biases entity/team (optional)

# set -euo pipefail

# export FAST_TRANSFORMER=O0
# export BYTED_TORCH_BYTECCL=O0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export OMP_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGTOK_ROOT="${REGTOK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/aofei}"
CKPT_ROOT="${CKPT_ROOT:-/data/aofei/output/MedSIGHT/checkpoints/align_stage1}"
REGTOK_WEIGHTS="${REGTOK_WEIGHTS:-/data/aofei/output/MedSight/Region_perceiver/0079280.pt}"
VISION_TOWER="${VISION_TOWER:-/data/aofei/CLIP/unimed_clip_vit_l14_base_text_encoder.pt}"
export HF_HUB_CACHE="/data/aofei/cache/huggingface"
export WANDB_PROJECT="${WANDB_PROJECT:-MedSIGHT_repo}"
# export WANDB_ENTITY="${WANDB_ENTITY:-}"

PRETRAIN_OUT_PATH="${CKPT_ROOT}"
PRETRAIN_TASK_NAME="$(basename "${PRETRAIN_OUT_PATH%/}")"

WORKER_NUM="${WORKER_NUM:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"


torchrun \
    --nnodes "$WORKER_NUM" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr localhost \
    --master_port 16668 \
    llava/train/train_mem.py \
        --model_name_or_path Qwen/Qwen3-8B \
        --version qwen \
        --data_path "${DATA_ROOT}/Medical_datasets/PubMedVision/Filter_PubMedVision_Alignment_VQA.json" \
        --image_folder "${DATA_ROOT}/Medical_datasets/PubMedVision" \
        --vision_tower "$VISION_TOWER" \
        --mm_vision_vq_type RegTok \
        --regtok_config_path "${REGTOK_ROOT}/source/regtok/regtok_config.yaml" \
        --regtok_weight_path "$REGTOK_WEIGHTS" \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --mm_vision_tuning_embedding False \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --bf16 True \
        --output_dir "$PRETRAIN_OUT_PATH" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --save_strategy steps \
        --save_steps 10000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing False \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name "$PRETRAIN_TASK_NAME" \
        --use_region_tokens True \
        --use_sep_proj False
