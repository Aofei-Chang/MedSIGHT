#!/bin/bash

export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"
export OMP_NUM_THREADS=8
export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt
export CUDA_VISIBLE_DEVICES="0"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--vq-ckpt "/qumulo/shared_data/aofei_summer/RegTok/source/RegTok_no_quant_abd/002-RegTok/checkpoints/0034160.pt" \
--data-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train" \
--segmentation-path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train_mask" \
--annotation-json "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/train.json" \
--num-stages 3 \
--finetune_codebook_only \
--finetune_decoder_only \
--beta2 0.99 \
--codebook-size 512 \
--codebook-embed-dim 64 \
--epochs 20 \
--global-batch-size 4 \
--ckpt-every 1000 \
--num-workers 0 \
--log-every 10 \
--use-quantization \
--quantization-loss-ratio 1 \
--up-sample-mode "query" \
--results-dir "RegTok_quant" \
--cloud-save-path "./logs/Reg-tok-training-quant/" \
"$@"
