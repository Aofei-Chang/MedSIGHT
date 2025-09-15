#!/bin/bash

export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"
export OMP_NUM_THREADS=8
export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt
# export CUDA_VISIBLE_DEVICES="4"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--batch-dataset-meta-file "/qumulo/shared_data/aofei_summer/data/BiomedParse_meta.json" \
--vq-ckpt "/qumulo/shared_data/aofei_summer/RegTok/source/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt" \
--num-stages 3 \
--finetune_codebook_only \
--finetune_decoder_only \
--beta2 0.99 \
--codebook-size 1024 \
--codebook-embed-dim 128 \
--num-queries 20 \
--epochs 4 \
--global-batch-size 4 \
--ckpt-every 10000 \
--num-workers 0 \
--log-every 10 \
--use-quantization \
--quantization-loss-ratio 1 \
--up-sample-mode "conv" \
--results-dir "RegTok_quant_full_test" \
--cloud-save-path "./logs/Reg-tok-quant-full/" \
"$@"
