"""Score multiple-choice VQA outputs (e.g. OmniMedVQA).

For each sample the prediction text is matched to the option whose string is
most similar to it; accuracy is the fraction of samples where this matches the
option chosen as ground truth.

Usage:
    python run_eval_mc.py \\
        --gt   /path/to/Modality_CT.json \\
        --pred /path/to/answers_CT.jsonl \\
        --eval_res /path/to/eval_CT.txt
"""

from __future__ import annotations

import argparse
import difflib
import json
import warnings

warnings.simplefilter('ignore')


def parse_args():
    parser = argparse.ArgumentParser('Multiple-choice evaluation for MedSIGHT outputs')
    parser.add_argument('--gt', required=True, type=str, help='Path to ground-truth JSON file.')
    parser.add_argument('--pred', required=True, type=str, help='Path to prediction JSONL file.')
    parser.add_argument('--eval_res', required=True, type=str, help='Path to write evaluation results.')
    return parser.parse_args()


def load_jsonl(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def str_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_most_similar_index(choices: list[str], target: str) -> int | None:
    best, best_idx = 0.0, None
    for i, choice in enumerate(choices):
        sim = str_similarity(choice, target)
        if sim > best:
            best, best_idx = sim, i
    return best_idx


def _gather_choices(sample: dict) -> list[str]:
    choices = [sample['option_A'], sample['option_B']]
    for letter in ('C', 'D'):
        key = f'option_{letter}'
        if key in sample:
            choices.append(sample[key])
    return choices


def _gt_value(sample: dict) -> str:
    if 'gt_answer' in sample:
        return sample['gt_answer']
    return sample['conversations'][1]['value'].lower()


def evaluate_mc(gt: list, pred: list) -> str:
    correct = total = 0
    for gt_sample, pred_sample in zip(gt, pred):
        choices = _gather_choices(gt_sample)
        pred_idx = find_most_similar_index(choices, pred_sample['text'])
        gt_idx = find_most_similar_index(choices, _gt_value(gt_sample))
        if pred_idx == gt_idx:
            correct += 1
        total += 1
    accuracy = correct / total if total else 0.0
    return f'Accuracy: {accuracy}'


def main():
    args = parse_args()

    dataset = args.gt.split('/')[-2]
    print(f'\n========\n {dataset}')

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    print(f'num_gt: {len(gt)} || num_pred: {len(pred)}')
    assert len(gt) == len(pred), 'gt and pred must be aligned 1:1'

    results = evaluate_mc(gt, pred)
    print(results)
    with open(args.eval_res, 'w') as f:
        f.write(results)


if __name__ == '__main__':
    main()
