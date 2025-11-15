#!/bin/bash

export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"
export OMP_NUM_THREADS=8
export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt
# export CUDA_VISIBLE_DEVICES="1"

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--batch-dataset-meta-file "/qumulo/shared_data/aofei_summer/data/BiomedParse_meta.json" \
--num-stages 3 \
--finetune_decoder_only \
--beta2 0.99 \
--codebook-size 2048 \
--num-queries 32 \
--codebook-embed-dim 64 \
--epochs 20 \
--global-batch-size 16 \
--ckpt-every 10000 \
--num-workers 0 \
--log-every 20 \
--use-quantization \
--quantization-loss-ratio 1 \
--up-sample-mode "conv" \
--results-dir "RegTok_pipeline_full" \
--cloud-save-path "./logs/Reg-tok-quant/" \
"$@"
