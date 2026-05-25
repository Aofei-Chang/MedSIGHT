"""Score open-ended VQA outputs against a ground-truth JSON file.

Usage:
    python run_eval.py \\
        --gt   /path/to/dataset/test.json \\
        --pred /path/to/answers.jsonl \\
        --eval_res /path/to/eval_results.txt

Use --modality_split to also break down metrics by the `modality` field of each
ground-truth item (currently used for BiomedParse-style splits).
"""

from __future__ import annotations

import argparse
import collections
import json
import warnings

from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate

from eval_metrics.evaluate_metrics import (
    calculate_exactmatch,
    calculate_f1score,
)
from eval_metrics.glossary import normalize_word

warnings.simplefilter('ignore')


def parse_args():
    parser = argparse.ArgumentParser('Evaluation for MedSIGHT generated outputs')
    parser.add_argument('--gt', required=True, type=str, help='Path to ground-truth JSON file.')
    parser.add_argument('--pred', required=True, type=str, help='Path to prediction JSONL file.')
    parser.add_argument('--eval_res', required=True, type=str, help='Path to write evaluation results.')
    parser.add_argument('--modality_split', action='store_true',
                        help='If set, additionally report metrics split by modality.')
    return parser.parse_args()


def load_jsonl(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def _extract_gt(item: dict) -> str:
    if 'conversations' in item:
        return item['conversations'][1]['value']
    if 'short_answer' in item:
        return item['short_answer']
    return item['answer']


def evaluate(gt: list, pred: list) -> str:
    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    num_close, num_open = 0, 0

    for gt_item, pred_item in zip(gt, pred):
        gt_value = normalize_word(_extract_gt(gt_item).lower())
        pred_value = normalize_word(pred_item['text'].lower())

        answer_type = gt_item['answer_type'].lower()
        if answer_type == 'open':
            num_open += 1
            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))

            f1, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)

            ref = [str(gt_value).lower().split()]
            hyp = str(pred_value).lower().split()
            bleu_scores['bleu_score'].append(sentence_bleu(ref, hyp))
            bleu_scores['bleu_score_1'].append(sentence_bleu(ref, hyp, weights=(1, 0, 0, 0)))
            bleu_scores['bleu_score_2'].append(sentence_bleu(ref, hyp, weights=(0, 1, 0, 0)))
            bleu_scores['bleu_score_3'].append(sentence_bleu(ref, hyp, weights=(0, 0, 1, 0)))

        elif answer_type.startswith('close'):
            num_close += 1
            if 'yes' in pred_value or 'no' in pred_value:
                closed_scores['hit'].append(int(gt_value in pred_value))
            else:
                closed_scores['hit'].append(0)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f'num_open {num_open} || num_close {num_close}')

    return tabulate(
        [
            ['exact match score', _mean(exact_scores['hit']) * 100],
            ['f1 score',          _mean(f1_scores['f1']) * 100],
            ['precision',         _mean(f1_scores['precision']) * 100],
            ['recall',            _mean(f1_scores['recall']) * 100],
            ['bleu_score',        _mean(bleu_scores['bleu_score']) * 100],
            ['bleu_score_1',      _mean(bleu_scores['bleu_score_1']) * 100],
            ['bleu_score_2',      _mean(bleu_scores['bleu_score_2']) * 100],
            ['bleu_score_3',      _mean(bleu_scores['bleu_score_3']) * 100],
            ['yes/no accuracy',   _mean(closed_scores['hit']) * 100],
        ],
        headers=['Metric', 'Performance'],
    )


def main():
    args = parse_args()

    dataset = args.gt.split('/')[-2]
    print(f'\n========\n {dataset}')

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    print(f'num_gt: {len(gt)} || num_pred: {len(pred)}')

    if args.modality_split:
        modality_data: dict[str, dict] = {}
        for gt_item, pred_item in zip(gt, pred):
            modality = gt_item.get('modality', 'unknown')
            if modality == 'Dermatoscopy':
                modality = 'Dermoscopy'
            bucket = modality_data.setdefault(modality, {'gt': [], 'pred': []})
            bucket['gt'].append(gt_item)
            bucket['pred'].append(pred_item)

        with open(args.eval_res, 'w') as f:
            for modality, data in modality_data.items():
                print(f'Evaluating modality: {modality}')
                results = evaluate(data['gt'], data['pred'])
                print(results)
                f.write(f'modality: {modality}\n{results}\n\n')

        results = evaluate(gt, pred)
        print('final results:', results)
        with open(args.eval_res, 'a') as f:
            f.write(results)
    else:
        results = evaluate(gt, pred)
        print(results)
        with open(args.eval_res, 'w') as f:
            f.write(results)


if __name__ == '__main__':
    main()
