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

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000   # enable flight recorder
# export NCCL_SOCKET_IFNAME=eth0  

export OMP_NUM_THREADS=8
export HF_HUB_CACHE="/data/aofei/cache/huggingface"
export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"

export CUDA_VISIBLE_DEVICES=1

export PRETRAIN_OUT_PATH=checkpoints/i2t_pre_region
export SFT_OUT_PATH=/data/aofei/output/MedSight/SLAKE/lora32_epoch6
export VISION_TOWER_CKPT="/data/aofei/CLIP/unimed_clip_vit_l14_base_text_encoder.pt"

PRETRAIN_TASK_NAME=$(basename "${PRETRAIN_OUT_PATH%/}")
SFT_TASK_NAME=$(basename "${SFT_OUT_PATH%/}")
WORKER_NUM=1
NPROC_PER_NODE=1

# torchrun \
# --nnodes $WORKER_NUM \
# --nproc_per_node $NPROC_PER_NODE \
# --master_addr localhost \
# --master_port 16668 \

# --peft_path /qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/reg_seg_105k_moe_sep_lora \
# --pretrained_llm_path /qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/i2t_instruct_region_final \
torchrun \
--nnodes $WORKER_NUM \
--nproc_per_node $NPROC_PER_NODE \
--master_addr localhost \
--master_port 16668 \
 llava/train/train_mem.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --pretrained_llm_path /data/aofei/output/MedSight/1110_full_instruct_71k_nosep \
    --version qwen \
    --data_path /data/aofei/hallucination/Slake/data/training.json \
    --image_folder /data/aofei/hallucination/Slake/imgs \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type RegTok \
    --regtok_config_path /home/avc6555/research/MedSight/RegTok/source/tokenizer/regtok_config.yaml \
    --regtok_weight_path /data/aofei/output/MedSight/Region_perceiver/0079280.pt \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_tuning_embedding False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir $SFT_OUT_PATH \
    --num_train_epochs 6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy steps \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name ${SFT_TASK_NAME} \
    --use_region_tokens True \
    --output_segmentation True \
    --use_seg_loss True \
    --modality_num 18 \
    --codebook_size 32 \
    --train_codebook False \
    --train_all_embeddings False \
    --resize_embedding False \
    --use_moe False \
    --moe_num_experts 0 \
    --align_regtok False \
    --final_tune_stage False \