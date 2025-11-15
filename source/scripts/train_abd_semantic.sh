#!/bin/bash

export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"
export OMP_NUM_THREADS=8
export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt
# export CUDA_VISIBLE_DEVICES="0"

root_dir="/qumulo/shared_data/aofei_summer/intern_records/RegTok"

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--data-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train" \
--segmentation-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train_mask" \
--annotation-json "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train.json" \
--num-stages 3 \
--finetune_decoder_only \
--beta2 0.99 \
--codebook-size 0 \
--num-queries 16 \
--codebook-embed-dim 0 \
--epochs 20 \
--global-batch-size 16 \
--ckpt-every 10000 \
--num-workers 0 \
--log-every 20 \
--quantization-loss-ratio 1 \
--up-sample-mode "conv" \
--use-semantic \
--results-dir "$root_dir/checkpoints/RegTok_pipeline_abd_semantic" \
--cloud-save-path "$root_dir/logs/logs/Reg-tok-abd-semantic/" \
"$@"
