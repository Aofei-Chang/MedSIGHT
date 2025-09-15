#!/bin/bash
# pip3 install -e . 
# pip3 install -e ".[train]" 

# CUR_DIR=$(cd `dirname $0`; pwd)

# cd ${CUR_DIR}/../..
export FAST_TRANSFORMER=O0
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

export OMP_NUM_THREADS=8
export HF_HUB_CACHE="/qumulo/shared_data/aofei_summer/LLMs"
export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"

export PRETRAIN_OUT_PATH=checkpoints/i2t_pre
export SFT_OUT_PATH=checkpoints/sft_slake
export VISION_TOWER_CKPT="/qumulo/shared_data/aofei_summer/CLIPs/unimed_clip_vit_l14.pt"

PRETRAIN_TASK_NAME=$(basename "${PRETRAIN_OUT_PATH%/}")
SFT_TASK_NAME=$(basename "${SFT_OUT_PATH%/}")
WORKER_NUM=1
NPROC_PER_NODE=1

# torchrun \
# --nnodes $WORKER_NUM \
# --nproc_per_node $NPROC_PER_NODE \
# --master_addr localhost \
# --master_port 16668 \
deepspeed llava/train/train_mem.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --version qwen \
    --deepspeed ./scripts/zero2.json \
    --pretrain_mm_mlp_adapter /qumulo/shared_data/aofei_summer/RegTok/RegLLM/checkpoints/i2t_pre/checkpoint-5053/mm_projector.bin \
    --data_path /qumulo/shared_data/aofei_summer/data/evaluation/training_masks.json \
    --image_folder /qumulo/shared_data/aofei_summer/data/evaluation/imgs \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type RegTok \
    --lora_enable True \
    --regtok_config_path /qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml \
    --regtok_weight_path /qumulo/shared_data/aofei_summer/RegTok/source/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_tuning_embedding False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $SFT_OUT_PATH \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${SFT_TASK_NAME} \
    --use_region_tokens False
