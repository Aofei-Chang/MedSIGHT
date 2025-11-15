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

export PRETRAIN_OUT_PATH=checkpoints/i2t_pre_region
export SFT_OUT_PATH=/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/ablation_1110_full_instruct_71k_no_stage1
export VISION_TOWER_CKPT="/qumulo/shared_data/aofei_summer/CLIPs/unimed_clip_vit_l14.pt"

PRETRAIN_TASK_NAME=$(basename "${PRETRAIN_OUT_PATH%/}")
SFT_TASK_NAME=$(basename "${SFT_OUT_PATH%/}")

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun \
# --nnodes $WORKER_NUM \
# --nproc_per_node $NPROC_PER_NODE \
# --master_addr localhost \
# --master_port 16668 \
deepspeed llava/train/train_mem.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --pretrained_llm_path /qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/reg_seg_align1110_nostage1 \
    --version qwen \
    --deepspeed ./scripts/zero3.json \
    --data_path /qumulo/shared_data/aofei_summer/data/RegAlign/final_data/Instruct_71k_1104.json \
    --image_folder /qumulo/shared_data/aofei_summer/data/LVLM/HuatuoGPT \
    --vision_tower $VISION_TOWER_CKPT \
    --mm_vision_vq_type RegTok \
    --regtok_config_path /qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml \
    --regtok_weight_path /qumulo/shared_data/aofei_summer/intern_records/RegTok/checkpoints/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_tuning_embedding False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $SFT_OUT_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy steps \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 24000 \
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
    --run_name ${SFT_TASK_NAME} \
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


    # --pretrain_mm_mlp_adapter /qumulo/shared_data/aofei_summer/RegTok/RegLLM/checkpoints/i2t_pre_region_final_sep/mm_projector.bin \