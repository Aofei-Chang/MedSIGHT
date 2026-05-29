#!/bin/bash
root_dir="root path of training results"
data_root_dir="the root path of data"
vq_ckpt_path="the checkpoint of previous pre-training stage of Region Perceiver"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--batch-dataset-meta-file "$data_root_dir/data/BiomedParse_meta.json" \
--vq-ckpt $vq_ckpt_path \
--num-stages 3 \
--finetune_codebook_only \
--finetune_decoder_only \
--beta2 0.99 \
--codebook-size 32 \
--codebook-embed-dim 64 \
--num-queries 20 \
--epochs 3 \
--global-batch-size 16 \
--ckpt-every 10000 \
--num-workers 0 \
--log-every 10 \
--use-quantization \
--num-modalities 18 \
--quantization-loss-ratio 1 \
--entropy-loss-ratio 0 \
--quant-use-seg \
--up-sample-mode "conv" \
--results-dir "$root_dir/checkpoints/RegTok_quant_full_modal" \
--cloud-save-path "$root_dir/logs/logs/Reg-tok-quant-full/" \
"$@"
