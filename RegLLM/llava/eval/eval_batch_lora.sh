cd /home/avc6555/research/MedSight/RegTok/RegLLM/llava/eval

# model_name="instruct_71k_1105_nosep"
model_name="lora16"

# python run_eval.py \
#         --gt /data/aofei/hallucination/VQA_RAD/data/test.json \
#         --pred /data/aofei/output/MedSight/VQA_RAD/inference/answers_${model_name}_epoch12.jsonl \
#         --eval_res /data/aofei/output/MedSight/VQA_RAD/inference/answers_${model_name}.txt

# python run_eval.py \
#         --gt /data/aofei/hallucination/Slake/data/test.json \
#         --pred /data/aofei/output/MedSight/SLAKE/inference/answers_${model_name}_epoch6.jsonl \
#         --eval_res /data/aofei/output/MedSight/SLAKE/inference/answers_${model_name}.txt

python run_eval.py \
        --gt /data/aofei/hallucination/PathVQA/pvqa/test.json \
        --pred /data/aofei/output/MedSight/PathVQA/inference/answers_${model_name}_epoch3.jsonl \
        --eval_res /data/aofei/output/MedSight/PathVQA/inference/answers_${model_name}_epoch3.txt
