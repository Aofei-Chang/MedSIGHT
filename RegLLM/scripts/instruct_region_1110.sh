#!/bin/bash
# Full-finetuning instruction tuning for MedSIGHT (Qwen3-8B + RegTok + segmentation).
#
# Resumes from the alignment checkpoint produced by `align_regtok_1110.sh` and
# fine-tunes the full model (LLM + projector + segmentation head) on the
# instruction-tuning corpus. For LoRA-only instruction tuning, see
# `instruct_regtok_lora_seg.sh`.
#
# Required paths (override via environment variables before running):
#   REGTOK_ROOT      – path to this repository
#   ALIGN_CKPT       – output dir of the alignment stage
#   DATA_ROOT        – root directory containing the instruction data
#   CKPT_ROOT        – directory to write checkpoints into
#   REGTOK_WEIGHTS   – pretrained Region Tokenizer weights
#   VISION_TOWER     – pretrained UniMed-CLIP weights
#   HF_HUB_CACHE     – HuggingFace cache

set -euo pipefail

export FAST_TRANSFORMER=O0
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGTOK_ROOT="${REGTOK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ALIGN_CKPT="${ALIGN_CKPT:-./checkpoints/reg_seg_align}"
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
CKPT_ROOT="${CKPT_ROOT:-./checkpoints}"
REGTOK_WEIGHTS="${REGTOK_WEIGHTS:-/path/to/RegTok/checkpoints/regtok_weights.pt}"
VISION_TOWER="${VISION_TOWER:-/path/to/CLIPs/unimed_clip_vit_l14.pt}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface}"

SFT_OUT_PATH="${CKPT_ROOT}/medsight_full_instruct"
SFT_TASK_NAME="$(basename "${SFT_OUT_PATH%/}")"

deepspeed llava/train/train_mem.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --version qwen \
    --deepspeed ./scripts/zero3.json \
    --pretrained_llm_path "$ALIGN_CKPT" \
    --data_path "${DATA_ROOT}/RegAlign/Instruct_71k.json" \
    --image_folder "${DATA_ROOT}/LVLM/HuatuoGPT" \
    --vision_tower "$VISION_TOWER" \
    --mm_vision_vq_type RegTok \
    --regtok_config_path "${REGTOK_ROOT}/source/regtok/regtok_config.yaml" \
    --regtok_weight_path "$REGTOK_WEIGHTS" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_tuning_embedding False \
    `# Selects the UniMed-CLIP block fed into the region perceiver. Must match the` \
    `# value in configs/model.yaml at inference time.` \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "$SFT_OUT_PATH" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy steps \
    --eval_steps 0 \
    --save_strategy steps \
    --save_steps 24000 \
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
    --run_name "$SFT_TASK_NAME" \
    --use_region_tokens True \
    --use_sep_proj False \
    --output_segmentation True \
    --use_seg_loss True \
    --modality_num 18 \
    --codebook_size 32 \
    --train_codebook True \
    --train_all_embeddings True \
    --resize_embedding False \
    --align_regtok True \
    --load_codebook_embeddings True \
    --use_lightweight_decoder False \
    --merge_codebook True
