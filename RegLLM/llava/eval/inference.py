"""Run MedSIGHT inference over a dataset configured in YAML.

Example:
    python -m llava.eval.inference \\
        --model-config configs/model.yaml \\
        --dataset-config configs/datasets.yaml \\
        --dataset VQA-RAD \\
        --run-name baseline

Outputs are written to {output_root}/{dataset}/answers_{run_name}.jsonl
(or, for OmniMedVQA, one file per modality).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import torch
import yaml
from tqdm import tqdm

from llava.eval.chatbot import RegLLMChatbot


def _get_nested(item: dict, dotted: str) -> Any:
    """Resolve a dotted path like 'conversations.0.value' against a dict/list."""
    value: Any = item
    for part in dotted.split('.'):
        if part.isdigit():
            value = value[int(part)]
        else:
            value = value[part]
    return value


def _load_questions(path: str) -> list[dict]:
    with open(os.path.expanduser(path), 'r') as f:
        return json.load(f)


def _process_split(
    bot: RegLLMChatbot,
    questions: list[dict],
    image_folder: Optional[str],
    answers_file: str,
    field_map: dict,
    prompt_suffix: str = "",
) -> None:
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as ans_file:
        for sample in tqdm(questions, desc=os.path.basename(answers_file)):
            idx = _get_nested(sample, field_map["id_field"])
            question = _get_nested(sample, field_map["question_field"])
            gt_ans = _get_nested(sample, field_map["answer_field"])
            image_rel = _get_nested(sample, field_map["image_field"])

            image_path = os.path.join(image_folder, image_rel) if image_folder else image_rel
            prompt = question + prompt_suffix

            with torch.inference_mode():
                ans = bot.inference(prompt, image_path)[0][0]
            ans = ans.replace("assistant\n", "").strip()

            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": prompt,
                "text": ans,
                "gt_ans": gt_ans,
                "metadata": {},
            }) + "\n")
            ans_file.flush()


def run_dataset(
    bot: RegLLMChatbot,
    dataset_name: str,
    dataset_cfg: dict,
    output_root: str,
    run_name: str,
) -> None:
    field_map = {
        "id_field": dataset_cfg["id_field"],
        "question_field": dataset_cfg["question_field"],
        "answer_field": dataset_cfg["answer_field"],
        "image_field": dataset_cfg["image_field"],
    }
    prompt_suffix = dataset_cfg.get("prompt_suffix", "")

    if "modalities" in dataset_cfg:
        # OmniMedVQA-style: one question file per modality.
        qa_root = dataset_cfg["qa_root"]
        image_folder = dataset_cfg.get("image_folder")
        for modality in dataset_cfg["modalities"]:
            question_file = os.path.join(qa_root, f"Modality_{modality}.json")
            answers_file = os.path.join(
                output_root, dataset_name, f"answers_{modality}_{run_name}.jsonl"
            )
            print(f"[{dataset_name}/{modality}] -> {answers_file}")
            questions = _load_questions(question_file)
            _process_split(bot, questions, image_folder, answers_file, field_map, prompt_suffix)
    else:
        question_file = dataset_cfg["question_file"]
        image_folder = dataset_cfg.get("image_folder")
        answers_file = os.path.join(output_root, dataset_name, f"answers_{run_name}.jsonl")
        print(f"[{dataset_name}] -> {answers_file}")
        questions = _load_questions(question_file)
        _process_split(bot, questions, image_folder, answers_file, field_map, prompt_suffix)


def main():
    parser = argparse.ArgumentParser(description="MedSIGHT batch inference")
    parser.add_argument("--model-config", required=True, help="Path to model YAML config.")
    parser.add_argument("--dataset-config", required=True, help="Path to datasets YAML config.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name, one of the keys under `datasets` in the YAML, or `all`.")
    parser.add_argument("--run-name", required=True,
                        help="Tag used in output filenames, e.g. `baseline` or `lora16`.")
    parser.add_argument("--output-root", default=None,
                        help="Override output_root from datasets.yaml.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.dataset_config, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    output_root = args.output_root or dataset_cfg.get("output_root", "./outputs/eval")

    bot = RegLLMChatbot.from_config(args.model_config, device=args.device)

    if args.dataset == "all":
        names = list(dataset_cfg["datasets"].keys())
    else:
        if args.dataset not in dataset_cfg["datasets"]:
            raise KeyError(
                f"Dataset '{args.dataset}' not found. "
                f"Available: {list(dataset_cfg['datasets'].keys())}"
            )
        names = [args.dataset]

    for name in names:
        run_dataset(bot, name, dataset_cfg["datasets"][name], output_root, args.run_name)


if __name__ == "__main__":
    main()
