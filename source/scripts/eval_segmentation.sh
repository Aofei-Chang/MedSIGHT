#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

python /qumulo/shared_data/aofei_summer/RegTok/source/evaluate_segmentation.py \
--ckpt_path "/qumulo/shared_data/aofei_summer/RegTok/source/RegTok_no_quant_abd/002-RegTok/checkpoints/0034160.pt" \
--data_path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/test" \
--mask_path "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/test_mask" \
--annotation_json "/qumulo/shared_data/aofei_summer/data/BiomedParse/amos22/amos22/CT/test.json" \
--batch_size 2 \
"$@"
