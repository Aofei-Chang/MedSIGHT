# python run_eval_mc.py \
#         --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_MRI.json \
#         --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_MRI_instruct_moe_r64.jsonl \
#         --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_MRI_instruct_moe_r64.txt \
#         --mc True

# python run_eval_mc.py \
#         --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_ultrasound.json \
#         --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_ultrasound_instruct_moe_r64.jsonl \
#         --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_ultrasound_instruct_moe_r64.txt \
#         --mc True


# python run_eval_mc.py \
#         --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_OCT.json \
#         --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_OCT_instruct_moe_r64.jsonl \
#         --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_OCT_instruct_moe_r64.txt \
#         --mc True

# python run_eval_mc.py \
#         --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_Microscopy.json \
#         --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_Microscopy_instruct_moe_r64.jsonl \
#         --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_Microscopy_instruct_moe_r64.txt \
#         --mc True

# python run_eval_mc.py \
#         --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_Fundus.json \
#         --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_Fundus_instruct_moe_r64.jsonl \
#         --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_Fundus_instruct_moe_r64.txt \
#         --mc True

# modalities=("CT" "OCT" "X-Ray" "MRI" "ultrasound" "Microscopy" "Fundus")
modalities=("Dermoscopy")
model_name="instruct_1105_lora16_12k"
for modality in "${modalities[@]}"; do
    python run_eval_mc.py \
            --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_${modality}.json \
            --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_${modality}_${model_name}.jsonl \
            --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_${modality}_${model_name}.txt \
            --mc True
done