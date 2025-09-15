python run_eval.py \
        --gt /qumulo/shared_data/aofei_summer/data/evaluation_data/VQA-RAD/test.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/VQA-RAD/inference/answers_instruct_region03.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/VQA-RAD/inference/eval_res_instruct_region03.txt

mv /qumulo/shared_data/aofei_summer/RegTok/RegLLM/logs /qumulo/shared_data/aofei_summer/intern_records/LVLM