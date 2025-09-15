#!/bin/bash

export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"
export OMP_NUM_THREADS=8
export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt
export CUDA_VISIBLE_DEVICES="1"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--data-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train" \
--segmentation-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train_mask" \
--annotation-json "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train.json" \
--finetune_decoder_only \
--num-stages 3 \
--beta2 0.95 \
--codebook-size 128 \
--codebook-embed-dim 16 \
--epochs 40 \
--global-batch-size 4 \
--ckpt-every 1000 \
--num-workers 0 \
--log-every 10 \
--use-quantization \
--quantization-loss-ratio 1 \
--results-dir "RegTok_quant" \
--cloud-save-path "./logs/Reg-tok-training-quant/" \
"$@"


# --use-quantization \