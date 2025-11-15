model_name="1105_full_306k_ins"

python run_eval.py \
        --gt /qumulo/shared_data/aofei_summer/data/evaluation/test_processed.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation/inference/answers_${model_name}.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation/inference/answers_${model_name}.txt