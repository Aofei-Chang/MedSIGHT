import os
import sys
dir_path = "/home/avc6555/research/MedSight/RegTok/RegLLM"
sys.path.insert(0, dir_path)
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["HF_HUB_CACHE"]="/data/aofei/cache/huggingface"
from llava.eval.cli_v1 import RegLLMChatbot
import torch
import json
from tqdm import tqdm

# peft_path = "/data/aofei/output/MedSight/SLAKE/lora32_epoch6/checkpoint-7380/"
# peft_path = "/data/aofei/output/MedSight/VQA-RAD/lora16_epoch8/checkpoint-2700/"
# peft_path = "/data/aofei/output/MedSight/VQA_RAD/lora32_epoch12/checkpoint-5400"
peft_path = "/data/aofei/output/MedSight/PathVQA/lora16_epoch3/checkpoint-14817"


model_dir = "/data/aofei/output/MedSight/1110_full_instruct_71k_nosep"
model_args = {
        "model_name_or_path": "Qwen/Qwen3-8B",
        "pretrained_llm_path": model_dir,
        "tokenizer_path": "/data/aofei/output/MedSight/1110_full_instruct_71k_nosep",
        "peft_path": peft_path,
        "regtok_config_path": "/home/avc6555/research/MedSight/RegTok/source/tokenizer/regtok_config.yaml",
        "regtok_weight_path": "/data/aofei/output/MedSight/Region_perceiver/0079280.pt",
        "use_regtok": True,
        "mm_vision_vq_type": "RegTok",
        "vision_tower": "/data/aofei/CLIP/unimed_clip_vit_l14_base_text_encoder.pt",
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
        "load_codebook_embeddings": False,
        "use_lightweight_decoder": False,
        "resize_embedding": False,
        "lora_r": 16,
        "lora_alpha": 32,
        # "lora_dropout": 0.05,
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
                idx = line["id"]
                question = line["conversations"][0]["value"] # ['value'].split('\n')[0]
                gt_ans = line['conversations'][1]['value'] # ['value']
                image_rel = line["image"]
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

    model_name = "lora16_epoch3"

    # 1.VQA-RAD
    # dataset_name = "VQA-RAD"
    # question_file = f"/data/aofei/hallucination/VQA_RAD/data/test.json"
    # answers_file = f"/data/aofei/output/MedSight/VQA_RAD/inference/answers_{model_name}.jsonl"
    # image_folder = f"/data/aofei/hallucination/VQA_RAD/images"
    # process_questions_file(question_file, image_folder, answers_file, dataset_name=dataset_name)

    # # # 2.SLAKE
    # dataset_name = "SLAKE"
    # question_file = "/data/aofei/hallucination/Slake/data/test.json"
    # answers_file = f"/data/aofei/output/MedSight/SLAKE/inference/answers_{model_name}.jsonl"
    # image_folder = "/data/aofei/hallucination/Slake/imgs"
    # process_questions_file(question_file, image_folder, answers_file, dataset_name=dataset_name)

    # # # 3.PathVQA
    dataset_name = "PathVQA"
    question_file = f"/data/aofei/hallucination/PathVQA/pvqa/test.json"
    answers_file = f"/data/aofei/output/MedSight/PathVQA/inference/answers_{model_name}.jsonl"
    image_folder = f"/data/aofei/hallucination/PathVQA/pvqa/images/test"
    process_questions_file(question_file, image_folder, answers_file, dataset_name=dataset_name)

    
if __name__ == "__main__":
    main()