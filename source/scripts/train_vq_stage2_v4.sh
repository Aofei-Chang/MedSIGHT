#!/bin/bash

export WANDB_API_KEY="10bcf42e60bfd806f25e11d5e055aa2c19ede264"
export OMP_NUM_THREADS=8
export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt
# export CUDA_VISIBLE_DEVICES="3"

root_dir="/qumulo/shared_data/aofei_summer/intern_records/RegTok"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
tokenizer/vq_train.py \
--dataset "biomed_seg" \
--batch-dataset-meta-file "/qumulo/shared_data/aofei_summer/data/BiomedParse_meta.json" \
--vq-ckpt "/qumulo/shared_data/aofei_summer/intern_records/RegTok/checkpoints/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt" \
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
--results-dir "$root_dir/checkpoints/RegTok_quant_full_modal_rec_seg_modal" \
--cloud-save-path "$root_dir/logs/logs/Reg-tok-quant-full-rec-seg-modal/" \
"$@"
