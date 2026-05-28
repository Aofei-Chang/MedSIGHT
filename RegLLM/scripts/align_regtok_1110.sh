#!/bin/bash
# Stage-1 alignment training for MedSIGHT (Qwen3-8B + RegTok + segmentation head).
#
# Trains the multimodal projector and segmentation alignment on top of a pretrained
# Region Tokenizer and the Qwen3-8B LLM. Outputs are written to $SFT_OUT_PATH.
#
# Required paths (override via environment variables before running):
#   REGTOK_ROOT     – path to this repository (defaults to current working dir)
#   DATA_ROOT       – root directory containing the alignment data
#   CKPT_ROOT       – directory to write checkpoints into
#   REGTOK_WEIGHTS  – pretrained Region Tokenizer weights
#   VISION_TOWER    – pretrained UniMed-CLIP weights
#   HF_HUB_CACHE    – HuggingFace cache (Qwen3 weights live here)

# set -euo pipefail

# export FAST_TRANSFORMER=O0
# export BYTED_TORCH_BYTECCL=O0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=lo
# export OMP_NUM_THREADS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGTOK_ROOT="${REGTOK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/aofei}"
CKPT_ROOT="${CKPT_ROOT:-/data/aofei/output/MedSIGHT/checkpoints/}"
REGTOK_WEIGHTS="${REGTOK_WEIGHTS:-/data/aofei/output/MedSight/Region_perceiver/0079280.pt}"
VISION_TOWER="${VISION_TOWER:-/data/aofei/CLIP/unimed_clip_vit_l14_base_text_encoder.pt}"
export HF_HUB_CACHE="/data/aofei/cache/huggingface"
export WANDB_PROJECT="${WANDB_PROJECT:-MedSIGHT_repo}"

PRETRAIN_OUT_PATH="/data/aofei/output/MedSIGHT/checkpoints/align_stage1"
SFT_OUT_PATH="${CKPT_ROOT}/reg_seg_align"
SFT_TASK_NAME="$(basename "${SFT_OUT_PATH%/}")"

WORKER_NUM="${WORKER_NUM:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

torchrun \
    --nnodes "$WORKER_NUM" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr localhost \
    --master_port 16668 \
    llava/train/train_mem.py \
        --model_name_or_path Qwen/Qwen3-8B \
        --pretrain_mm_mlp_adapter "${PRETRAIN_OUT_PATH}/mm_projector.bin" \
        --version qwen \
        --data_path "${DATA_ROOT}/Medical_datasets/PubMedVision/RegAlign/Seg_instruct_63k_updated.json" \
        --image_folder "${DATA_ROOT}/Medical_datasets/PubMedVision" \
        --codebook_path "${DATA_ROOT}/Medical_datasets/PubMedVision/RegAlign/projected_codebook.pt" \
        --vision_tower "$VISION_TOWER" \
        --mm_vision_vq_type RegTok \
        --regtok_config_path "${REGTOK_ROOT}/source/regtok/regtok_config.yaml" \
        --regtok_weight_path "$REGTOK_WEIGHTS" \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter False \
        --mm_vision_tuning_embedding False \
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
        --learning_rate 5e-4 \
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
        --run_name "$SFT_TASK_NAME" \
        --use_region_tokens True \
        --output_segmentation True \
        --use_seg_loss True \
        --modality_num 18 \
        --codebook_size 32 \
        --train_codebook True \
        --train_all_embeddings False \
        --align_regtok True \
        --seg_align_stage True \
        --use_sep_proj False \
        --use_lightweight_decoder False
