import os
import sys
dir_path = "/qumulo/shared_data/aofei_summer/RegTok/RegLLM"
sys.path.insert(0, dir_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["HF_HUB_CACHE"]="/qumulo/shared_data/aofei_summer/LLMs"
from llava.eval.cli_v1 import RegLLMChatbot


# model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/i2t_instruct_region_final"
# model_args = {
#         "model_name_or_path": "Qwen/Qwen3-8B",
#         "pretrained_llm_path": model_dir,
#         "tokenizer_path": "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/reg_seg_instruct_mix_c_45k",
#         "peft_path": "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/reg_seg_83k_lora32",
#         "regtok_config_path": "/qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml",
#         "regtok_weight_path": "/qumulo/shared_data/aofei_summer/intern_records/RegTok/checkpoints/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt",
#         "use_regtok": True,
#         "mm_vision_vq_type": "RegTok",
#         "use_region_tokens": False,
#         "vision_tower": "/qumulo/shared_data/aofei_summer/CLIPs/unimed_clip_vit_l14.pt",
#         "mm_use_im_start_end": False,
#         "mm_use_im_patch_token": True,
#         "mm_vision_select_feature": "patch",
#         "mm_patch_merge_type": "flat",
#         "mm_projector_type": "mlp2x_gelu",
#         "pretrain_mm_mlp_adapter": None,
#         "mm_vision_select_layer": -1,
#         "use_region_tokens": True,
#         "use_sep_proj": False,
#         "output_segmentation": True,
#         "modality_num": 18,
#         "codebook_size": 32,
#         "train_all_embeddings": False,
#         "use_moe": False
#     }

model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/reg_seg_align1102"
model_args = {
        "model_name_or_path": "Qwen/Qwen3-8B",
        "pretrained_llm_path": model_dir,
        "tokenizer_path": "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/reg_seg_align1102",
        "peft_path": "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/1103_lora16_71k_epoch3",
        # "peft_path": "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/new_lora128_moe_390k",
        "regtok_config_path": "/qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml",
        "regtok_weight_path": "/qumulo/shared_data/aofei_summer/intern_records/RegTok/checkpoints/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt",
        "use_regtok": True,
        "mm_vision_vq_type": "RegTok",
        "use_region_tokens": False,
        "vision_tower": "/qumulo/shared_data/aofei_summer/CLIPs/unimed_clip_vit_l14.pt",
        "mm_use_im_start_end": False,
        "mm_use_im_patch_token": True,
        "mm_vision_select_feature": "patch",
        "mm_patch_merge_type": "flat",
        "mm_projector_type": "mlp2x_gelu",
        "pretrain_mm_mlp_adapter": None,
        "mm_vision_select_layer": -1,
        "use_region_tokens": True,
        "use_sep_proj": False,
        "use_seg_loss": True,
        "output_segmentation": True,
        "modality_num": 18,
        "codebook_size": 32,
        "train_all_embeddings": False,
        "load_codebook_embeddings": True,
        "resize_embedding": False,
        "use_lightweight_decoder": False,
        "lora_r": 16,
        "use_moe": False,
    }

bot = RegLLMChatbot(model_dir, model_args=model_args, device="cuda")


import json
from tqdm import tqdm
model_name = "instruct_lora16_71k"
# dataset_name = "VQA-RAD"
dataset_name = "OmniMed"

modality_name = "Dermoscopy"
question_file = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_Dermoscopy.json"
answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{modality_name}_{model_name}.jsonl"
image_folder = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA"


questions = json.load(open(os.path.expanduser(question_file), "r"))
answers_file = os.path.expanduser(answers_file)
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")
for line in tqdm(questions):


    idx = line["question_id"]
    question = line["question"] + ". Answer this question shortly by only selecting one option." # ['value'].split('\n')[0]
    gt_ans = line["gt_answer"] # ['value']      
    image_file = line["image_path"]
    qs = question
    
    image_file = os.path.join(image_folder, image_file)
    ans = bot.inference(qs, image_file)[0][0]
    ans = ans.replace("assistant\n", "").strip()

    ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": ans,
                                   "gt_ans": gt_ans,
                                   "metadata": {}}) + "\n")
    ans_file.flush()
ans_file.close()


modality_name = "CT"
question_file = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_CT.json"
answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{modality_name}_{model_name}.jsonl"
image_folder = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA"


questions = json.load(open(os.path.expanduser(question_file), "r"))
answers_file = os.path.expanduser(answers_file)
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")
for line in tqdm(questions):


    idx = line["question_id"]
    question = line["question"] + ". Answer this question shortly by only selecting one option." # ['value'].split('\n')[0]
    gt_ans = line["gt_answer"] # ['value']      
    image_file = line["image_path"]
    qs = question

    image_file = os.path.join(image_folder, image_file)
    ans = bot.inference(qs, image_file)[0]
    ans = ans.replace("assistant\n", "").strip()

    ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": ans,
                                   "gt_ans": gt_ans,
                                   "metadata": {}}) + "\n")
    ans_file.flush()
ans_file.close()


modality_name = "OCT"
question_file = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_OCT.json"
answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{modality_name}_{model_name}.jsonl"
image_folder = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA"


questions = json.load(open(os.path.expanduser(question_file), "r"))
answers_file = os.path.expanduser(answers_file)
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")
for line in tqdm(questions):


    idx = line["question_id"]
    question = line["question"] + ". Answer this question shortly by only selecting one option." # ['value'].split('\n')[0]
    gt_ans = line["gt_answer"] # ['value']      
    image_file = line["image_path"]
    qs = question
    
    image_file = os.path.join(image_folder, image_file)
    ans = bot.inference(qs, image_file)[0]
    ans = ans.replace("assistant\n", "").strip()

    ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": ans,
                                   "gt_ans": gt_ans,
                                   "metadata": {}}) + "\n")
    ans_file.flush()
ans_file.close()

# modality_name = "XRay"
# question_file = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_X-Ray.json"
# answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{modality_name}_{model_name}.jsonl"
# image_folder = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA"


# questions = json.load(open(os.path.expanduser(question_file), "r"))
# answers_file = os.path.expanduser(answers_file)
# os.makedirs(os.path.dirname(answers_file), exist_ok=True)
# ans_file = open(answers_file, "w")
# for line in tqdm(questions):


#     idx = line["question_id"]
#     question = line["question"] + ". Answer this question shortly by only selecting one option." # ['value'].split('\n')[0]
#     gt_ans = line["gt_answer"] # ['value']      
#     image_file = line["image_path"]
#     qs = question
    
#     image_file = os.path.join(image_folder, image_file)
#     ans = bot.inference(qs, image_file)[0]
#     ans = ans.replace("assistant\n", "").strip()

#     ans_file.write(json.dumps({"question_id": idx,
#                                    "prompt": qs,
#                                    "text": ans,
#                                    "gt_ans": gt_ans,
#                                    "metadata": {}}) + "\n")
#     ans_file.flush()
# ans_file.close()


# modality_name = "MRI"
# question_file = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access/Modality_MR (Mag-netic Resonance Imaging).json"
# answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{modality_name}_{model_name}.jsonl"
# image_folder = f"/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA"


# questions = json.load(open(os.path.expanduser(question_file), "r"))
# answers_file = os.path.expanduser(answers_file)
# os.makedirs(os.path.dirname(answers_file), exist_ok=True)
# ans_file = open(answers_file, "w")
# for line in tqdm(questions):


#     idx = line["question_id"]
#     question = line["question"] + ". Answer this question shortly by only selecting one option." # ['value'].split('\n')[0]
#     gt_ans = line["gt_answer"] # ['value']      
#     image_file = line["image_path"]
#     qs = question
    
#     image_file = os.path.join(image_folder, image_file)
#     ans = bot.inference(qs, image_file)[0]
#     ans = ans.replace("assistant\n", "").strip()

#     ans_file.write(json.dumps({"question_id": idx,
#                                    "prompt": qs,
#                                    "text": ans,
#                                    "gt_ans": gt_ans,
#                                    "metadata": {}}) + "\n")
#     ans_file.flush()
# ans_file.close()