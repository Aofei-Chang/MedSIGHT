import argparse
import json
import collections
import random
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from eval_metrics.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from eval_metrics.glossary import *


import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--candidate', type=str, default="candidate.json", help='path to candidate answer file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--eval_res', type=str, default="eval_res.txt", help='path to prediction file', )
    parser.add_argument('--modality_split', type=bool, default=False, help='whether to split evaluation by modality')
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred, candidate, criterion=None):    
    closed_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    # open_hit_scores = collections.defaultdict(list)
    num_close, num_open = 0, 0
    for gt_item, pred_item in zip(gt, pred):
        if "conversations" in gt_item:
            gt_value = gt_item['conversations'][1]['value'].lower()
        else:
            if "short_answer" in gt_item:
                gt_value = gt_item["short_answer"].lower()
            else:
                gt_value = gt_item["answer"].lower()
        pred_value = pred_item['text'].lower()

        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        if gt_item['answer_type'].lower() == 'open':
            num_open += 1
            # for open-ended question
            # if gt_value in pred_value:
            #     hit = 1.0
            # else:
            #     hit = 0.0
            # open_hit_scores['hit'].append(hit)

            

            # open_hit_scores['hit'].append(calculate_appearance_with_normalization(pred_value, gt_value, candidate))
            # open_hit_scores['q_id'].append(pred_item['question_id'])

            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(pred_item['question_id'])


            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(pred_item['question_id'])

            # if isinstance(f1_scores['hit'][-1], str):
            #     # import pdb; pdb.set_trace()

            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
            
            bleu_scores['q_id'].append(pred_item['question_id'])
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)

        elif gt_item['answer_type'].lower().startswith('close'):
            # for close-ended question (Yes/No)
            num_close += 1
            closed_scores['q_id'].append(pred_item['question_id'])
            if 'yes' in pred_value or 'no' in pred_value:
                if gt_value in pred_value:
                    closed_scores['hit'].append(1)
                else:
                    closed_scores['hit'].append(0)
            else:
                closed_scores['hit'].append(0)
    
    # import pdb; pdb.set_trace()
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
    precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
    recall = sum(f1_scores['recall']) / len(f1_scores['recall'])

    bleu_score   = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score'])
    bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1'])
    bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2'])
    bleu_score_3 = sum(bleu_scores['bleu_score_3']) / len(bleu_scores['bleu_score_3'])

    # open_hit_score = sum(open_hit_scores['hit']) / len(open_hit_scores['hit'])
    closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0

    # num_open, num_close = len(closed_scores['hit']), len(open_hit_scores['hit'])
    print(f'num_open {num_open} || num_close {num_close}')

    return tabulate(
        [
            ['exact match score', exact_score*100], 
            ['f1 score', f1_score*100], 
            ['precision', precision*100], 
            ['recall', recall*100], 
            ['bleu_score', bleu_score*100], 
            ['bleu_score_1', bleu_score_1*100], 
            ['bleu_score_2', bleu_score_2*100], 
            ['bleu_score_3', bleu_score_3*100], 
            # ['open accuracy', open_hit_score*100],
            ['yes/no accuracy', closed_score*100]
        ], 
        headers=['Metric', 'Performance']
    )

if __name__ == '__main__':
    args = parse_option()

    dataset = args.gt.split("/")[-2]
    print(f"\n========\n {dataset}")

    gt = json.load(open(args.gt, 'r'))
    # candidate = json.load(open(args.candidate, 'r'))
    pred = load_jsonl(args.pred)

    # gt_ids = [item['qid'] for item in gt if "qid" in item]
    gt_ids = []
    for item in gt:
        if "qid" in item:
            gt_ids.append(item['qid'])
        elif "id" in item:
            gt_ids.append(item['id'])
        elif "qa_id" in item:
            gt_ids.append(item['qa_id'])
    pred_ids = [item['question_id'] for item in pred]
    # filter pred_ids not in gt_ids
    # pred_ids = [id for id in pred_ids if id in gt_ids]

    num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
    print(f'num_gt_ids: {num_gt_ids} || num_pred_ids: {num_pred_ids}')
    # import pdb; pdb.set_trace()
    # assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    if args.modality_split:
        # perform evaluation with modality split
        modality_data = {}
        for i in range(len(gt)):
            gt_i = gt[i]
            pred_i = pred[i]
            modality_i = gt_i.get("modality", "unknown")
            if modality_i == "Dermatoscopy":
                modality_i = "Dermoscopy"
            if modality_i not in modality_data:
                modality_data[modality_i] = {"gt": [], "pred": []}
            modality_data[modality_i]["gt"].append(gt_i)
            modality_data[modality_i]["pred"].append(pred_i)
        with open(f"{args.eval_res}", "w") as f:
            for modality, data in modality_data.items():
                print(f"Evaluating modality: {modality}")
                results = evaluate(data["gt"], data["pred"], candidate=None)
                print(results)
                f.write(f"modality: {modality}\n")
                f.write(results)
            f.close()
        
        # perform evaluation
        results = evaluate(gt, pred, candidate=None)
        print("final results:", results)
        with open(args.eval_res, "a") as f:
            f.write(results)
            f.close()
        
    else:
        # perform evaluation
        results = evaluate(gt, pred, candidate=None)
        print("final results:", results)
        with open(args.eval_res, "w") as f:
            f.write(results)
            f.close()