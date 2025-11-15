python inference_batch_ablation.py

cd /qumulo/shared_data/aofei_summer/RegTok/RegLLM/llava/eval

# model_name="ablation_71k_1105_noregion"
model_name="ablation_71k_1105_nostage1"
# model_name="ablation_71k_1105_nostage2"
# model_name="ablation_71k_1105_nostage3"

python run_eval.py \
        --gt /qumulo/shared_data/aofei_summer/data/evaluation_data/VQA-RAD/test.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/VQA-RAD/inference/answers_${model_name}.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/VQA-RAD/inference/answers_${model_name}.txt

python run_eval.py \
        --gt /qumulo/shared_data/aofei_summer/data/evaluation/test_processed.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation/inference/answers_${model_name}.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation/inference/answers_${model_name}.txt

python run_eval.py \
        --gt /qumulo/shared_data/aofei_summer/data/PathVQA/pvqa/test.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/PathVQA/inference/answers_${model_name}.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/PathVQA/inference/answers_${model_name}.txt

python run_eval.py \
        --gt /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/SegVQA_Diagnostic_test_vqa.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/answers_${model_name}.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/inference/answers_${model_name}.txt \
        --modality_split True

# modalities=("CT" "OCT" "X-Ray" "MRI" "ultrasound" "Microscopy" "Fundus")
# for modality in "${modalities[@]}"; do
#     python run_eval_mc.py \
#             --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_${modality}.json \
#             --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_${modality}_${model_name}.jsonl \
#             --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_${modality}_${model_name}.txt \
#             --mc True
# done