#!/bin/bash
# pip3 install -e . 
# pip3 install -e ".[train]" 

# CUR_DIR=$(cd `dirname $0`; pwd)

# cd ${CUR_DIR}/../..
# export FAST_TRANSFORMER=O0
# export BYTED_TORCH_BYTECCL=O0
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0

export OMP_NUM_THREADS=8
export HF_HUB_CACHE="/data/aofei/cache/huggingface"
export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"

# export CUDA_VISIBLE_DEVICES="1"

export PRETRAIN_OUT_PATH=checkpoints/i2t_pre_region_0326
export SFT_OUT_PATH=checkpoints/i2t_sft
export VISION_TOWER_CKPT="/data/aofei/CLIP/unimed_clip_vit_l14_base_text_encoder.pt"

PRETRAIN_TASK_NAME=$(basename "${PRETRAIN_OUT_PATH%/}")
SFT_TASK_NAME=$(basename "${SFT_OUT_PATH%/}")
WORKER_NUM=1
NPROC_PER_NODE=1
### pretrain ####

    # --deepspeed ./scripts/zero2.json \

    # --data_path /qumulo/shared_data/aofei_summer/data/LVLM/data/alignment/llava_med_alignment_100k.json \
    # --image_folder /qumulo/shared_data/aofei_summer/data/LVLM/data/images \
torchrun \
--nnodes $WORKER_NUM \
--nproc_per_node $NPROC_PER_NODE \
--master_addr localhost \
--master_port 16668 \
    llava/train/train_mem.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --version qwen \
    --data_path /data/aofei/hallucination/Slake/data/training.json \
    --image_folder /data/aofei/hallucination/Slake/imgs/ \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type RegTok \
    --regtok_config_path /home/avc6555/research/MedSight/RegTok/source/tokenizer/regtok_config.yaml \
    --regtok_weight_path /data/aofei/output/MedSight/Region_perceiver/0079280.pt \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_tuning_embedding False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $PRETRAIN_OUT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --run_name ${PRETRAIN_TASK_NAME} \
    --use_region_tokens True \
    --use_sep_proj False
