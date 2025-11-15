python run_eval_mc.py \
        --gt /qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_OCT.json \
        --pred /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/answers_OCT_instruct_regseg_lora_83k.jsonl \
        --eval_res /qumulo/shared_data/aofei_summer/data/evaluation_data/OmniMed/inference/eval_OCT_regseg_lora_83k.txt \
        --mc True