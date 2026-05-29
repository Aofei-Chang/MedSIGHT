#!/bin/bash

data_root_dir="the root path of data"

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--batch-dataset-meta-file "$data_root_dir/data/BiomedParse_meta.json" \
--num-stages 3 \
--finetune_decoder_only \
--beta2 0.99 \
--codebook-size 0 \
--num-queries 20 \
--codebook-embed-dim 0 \
--epochs 20 \
--global-batch-size 48 \
--ckpt-every 10000 \
--num-workers 0 \
--log-every 20 \
--quantization-loss-ratio 1 \
--up-sample-mode "conv" \
--results-dir "RegTok_pipeline_full_wo_quant" \
--cloud-save-path "./logs/Reg-tok-wo-quant/" \
"$@"

