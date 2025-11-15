import os
import sys
dir_path = "/qumulo/shared_data/aofei_summer/RegTok/RegLLM"
sys.path.insert(0, dir_path)
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["HF_HUB_CACHE"]="/qumulo/shared_data/aofei_summer/LLMs"
from llava.eval.cli_v1 import RegLLMChatbot
import torch
import json
from tqdm import tqdm


# model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/ablation_noregion_71k_nosep"
# model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/1107_ablation_71k_seg_token"
model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/ablation_1110_full_instruct_71k_no_stage1"
# model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/ablation_1110_full_instruct_71k_no_stage2"
# model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/ablation_1110_full_instruct_71k_no_stage3"
# model_dir = "/qumulo/shared_data/aofei_summer/intern_records/LVLM/checkpoints/reg_seg_align1110"
# model_args = {
#         "model_name_or_path": "Qwen/Qwen3-8B",
#         "pretrained_llm_path": model_dir,
#         "tokenizer_path": model_dir,
#         "peft_path": None,
#         "regtok_config_path": "/qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml",
#         "regtok_weight_path": "/qumulo/shared_data/aofei_summer/intern_records/RegTok/checkpoints/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt",
#         "use_regtok": True,
#         "mm_vision_vq_type": "RegTok",
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
#         "use_seg_loss": True,
#         "output_segmentation": True,
#         "modality_num": 18,
#         "codebook_size": 32,
#         "train_all_embeddings": True,
#         "load_codebook_embeddings": True,
#         "use_lightweight_decoder": False,
#         "resize_embedding": False,
#         # "lora_r": 64,
#         # "lora_alpha": 64,
#         # # "lora_dropout": 0.05,
#         "use_moe": False,

#     }

model_args = {
        "model_name_or_path": "Qwen/Qwen3-8B",
        "pretrained_llm_path": model_dir,
        "tokenizer_path": model_dir,
        "peft_path": None,
        "regtok_config_path": "/qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml",
        "regtok_weight_path": "/qumulo/shared_data/aofei_summer/intern_records/RegTok/checkpoints/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt",
        "use_regtok": True,
        "mm_vision_vq_type": "RegTok",
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
        "train_all_embeddings": True,
        "load_codebook_embeddings": False,
        "use_lightweight_decoder": False,
        "resize_embedding": False,
        # "lora_r": 64,
        # "lora_alpha": 64,
        # # "lora_dropout": 0.05,
        "use_moe": False,

    }

bot = RegLLMChatbot(model_dir, model_args=model_args, device="cuda")

@torch.inference_mode()
def generate_answer(image_file, qs) -> str:
    ans = bot.inference(qs, image_file)[0][0]
    ans = ans.replace("assistant\n", "").strip()
    return ans



def process_questions_file(question_file: str, image_folder: str, answers_file: str, dataset_name=""):
    questions = json.load(open(os.path.expanduser(question_file), "r"))
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            if dataset_name == "OmniMed":
                idx = line["question_id"]
                gt_ans = line["gt_answer"]
                image_rel = line["image_path"]
            elif dataset_name == "BiomedParse":
                image_rel = line["image_file"]
                idx = line["qa_id"]
                gt_ans = line["short_answer"]
                question = line["question"]
            elif dataset_name == "PathVQA":
                idx = line["id"]
                question = line["conversations"][0]['value'] # ['value'].split('\n')[0]
                gt_ans = line["conversations"][1]['value']  
                image_rel = line["image"]
            elif dataset_name == "SLAKE":
                idx = line["qid"]
                question = line["question"] # ['value'].split('\n')[0]
                gt_ans = line["answer"] # ['value']      
                image_rel = line["img_name"]
            elif dataset_name == "VQA-RAD":
                idx = line["id"]
                question = line["conversations"][0]["value"] # ['value'].split('\n')[0]
                gt_ans = line['conversations'][1]['value'] # ['value']
                image_rel = line["image"]

            if image_folder:
                image_path = os.path.join(image_folder, image_rel)
            else:
                image_path = image_rel
            if dataset_name == "OmniMed":
                question = line["question"] + ". Answer this question shortly by only selecting one option."
            # else:
            #     question += "Answer this question concisely."
            ans = generate_answer(image_path, question)

            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": question,
                "text": ans,
                "gt_ans": gt_ans,
                "metadata": {}
            }) + "\n")
            ans_file.flush()




def main():

    # model_name = "ablation_71k_1105_noregion"
    # model_name = "ablation_71k_1105_nocodebook"
    # model_name = "ablation_71k_1105_nostage2"
    # model_name = "ablation_71k_1105_nostage3"
    model_name = "ablation_71k_1105_nostage1"

    # 1.VQA-RAD
    dataset_name = "VQA-RAD"
    question_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/test.json"
    answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{model_name}.jsonl"
    image_folder = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/image"
    process_questions_file(question_file, image_folder, answers_file, dataset_name=dataset_name)

    # # # 2.SLAKE
    dataset_name = "SLAKE"
    question_file = "/qumulo/shared_data/aofei_summer/data/evaluation/test_processed.json"
    answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation/inference/answers_{model_name}.jsonl"
    image_folder = "/qumulo/shared_data/aofei_summer/data/evaluation/imgs"
    process_questions_file(question_file, image_folder, answers_file, dataset_name=dataset_name)

    # # # 3.PathVQA
    dataset_name = "PathVQA"
    question_file = f"/qumulo/shared_data/aofei_summer/data/PathVQA/pvqa/test.json"
    answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{model_name}.jsonl"
    image_folder = f"/qumulo/shared_data/aofei_summer/data/PathVQA/pvqa/images/test"
    process_questions_file(question_file, image_folder, answers_file, dataset_name=dataset_name)

    # # 4. MeDSeg
    dataset_name = "BiomedParse"
    question_file = "/qumulo/shared_data/aofei_summer/data/evaluation_data/BiomedParse/SegVQA_Diagnostic_test_vqa.json"
    answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{model_name}.jsonl"
    image_folder = None
    process_questions_file(question_file, image_folder, answers_file, dataset_name=dataset_name)

    # # 5.OmniMed

    # dataset_name = "OmniMed"

    # base_qa_dir = "/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA/QA_information/Open-access"
    # image_folder = "/qumulo/shared_data/aofei_summer/data/OmniMed/OmniMedVQA/OmniMedVQA"

    # Modalities to process
    # modalities = ["CT", "OCT", "X-Ray", "MRI", "ultrasound", "Microscopy", "Fundus"]

    # for modality in modalities:
    #     question_file = f"{base_qa_dir}/Modality_{modality}.json"
    #     answers_file = f"/qumulo/shared_data/aofei_summer/data/evaluation_data/{dataset_name}/inference/answers_{modality}_{model_name}.jsonl"
    #     process_questions_file(question_file, image_folder, answers_file, dataset_name="OmniMed")


if __name__ == "__main__":
    main()