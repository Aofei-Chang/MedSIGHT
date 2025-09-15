#!/bin/bash

export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"
export OMP_NUM_THREADS=8
# export CUDA_VISIBLE_DEVICES="2,3"

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--data-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train" \
--segmentation-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train_mask" \
--annotation-json "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train.json" \
--num-stages 4 \
--codebook-size 512 \
--epochs 40 \
--global-batch-size 6 \
--ckpt-every 1000 \
--num-workers 0 \
--log-every 10 \
--cloud-save-path "./logs/Reg-tok-training/" \
"$@"

# --use-quantization \