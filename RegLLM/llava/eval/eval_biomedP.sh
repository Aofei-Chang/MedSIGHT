# python run_eval.py \
#         --gt /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/SegVQA_Diagnostic_test.json \
#         --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/answers_Qwen2_5_7B_ins_long.jsonl \
#         --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/answers_Qwen2_5_7B_ins_long.txt \
#         --modality_split True

# mv /qumulo/shared_data/aofei_summer/RegTok/RegLLM/logs /qumulo/shared_data/aofei_summer/intern_records/LVLM
#

# python run_eval.py \
#         --gt /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/SegVQA_Diagnostic_test_v2_filtered.json \
#         --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/PLIB_answers_vqa.jsonl \
#         --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/PLIB_answers_vqa.txt \
#         --modality_split True

python run_eval.py \
        --gt /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/SegVQA_Diagnostic_test_v2_filtered.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/LaSagnA_answers_vqa.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/LaSagnA_answers_vqa.txt \
        --modality_split True