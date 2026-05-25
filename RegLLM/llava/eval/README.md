# MedSIGHT Evaluation

Inference and scoring tools for the **MedSIGHT** medical large vision-language model.

```
llava/eval/
├── chatbot.py          # RegLLMChatbot — the inference API
├── inference.py        # Batch inference driven by YAML configs
├── run_eval.py         # Score open-ended VQA outputs
├── run_eval_mc.py      # Score multiple-choice VQA outputs (OmniMedVQA)
├── configs/
│   ├── model.yaml      # Model checkpoint + architecture flags
│   └── datasets.yaml   # Dataset paths and field mappings
├── scripts/            # Thin shell wrappers (run_inference.sh, eval_*.sh)
├── examples/
│   ├── vqa_rad_demo.ipynb
│   └── segmentation_demo.ipynb
└── eval_metrics/       # BLEU / F1 / exact-match metric helpers
```

## 1. Setup

Edit `configs/model.yaml` to point to your local MedSIGHT checkpoint, RegTok
weights, and UniMed-CLIP vision encoder. The defaults are placeholders.

Edit `configs/datasets.yaml` to point at the test splits you want to evaluate.
The shipped entries cover the five datasets used in the paper:

| Dataset       | Type             | Notes                                        |
|---------------|------------------|----------------------------------------------|
| VQA-RAD       | open-ended VQA   | radiology                                    |
| SLAKE         | open-ended VQA   | bilingual, English split                     |
| PathVQA       | open-ended VQA   | pathology                                    |
| OmniMedVQA    | multiple-choice  | 7 modalities, evaluated per-modality         |
| BiomedParse   | open-ended + seg | segmentation-augmented VQA                   |

## 2. Single-turn inference (Python API)

```python
from llava.eval.chatbot import RegLLMChatbot

bot = RegLLMChatbot.from_config("configs/model.yaml", device="cuda")

answers, _ = bot.inference("What modality is used to take this image?",
                           "/path/to/image.jpg")
print(answers[0])
```

For segmentation (requires `output_segmentation: true` in `model.yaml`):

```python
result = bot.inference("Please segment the kidney.",
                       "/path/to/image.jpg", output_seg=True)
print(result["answers"][0])
print(result["mask_logits"].shape)   # (B, K, H, W)
```

See [`examples/vqa_rad_demo.ipynb`](examples/vqa_rad_demo.ipynb) and
[`examples/segmentation_demo.ipynb`](examples/segmentation_demo.ipynb).

## 3. Batch inference

Run inference over a configured dataset and write predictions to JSONL:

```bash
# Single dataset
python -m llava.eval.inference \
    --model-config   configs/model.yaml \
    --dataset-config configs/datasets.yaml \
    --dataset        VQA-RAD \
    --run-name       baseline

# Or, every dataset listed in datasets.yaml
python -m llava.eval.inference \
    --model-config   configs/model.yaml \
    --dataset-config configs/datasets.yaml \
    --dataset        all \
    --run-name       baseline
```

The shell wrapper does the same thing:

```bash
scripts/run_inference.sh VQA-RAD baseline
scripts/run_inference.sh all     baseline
```

Predictions are written to
`{output_root}/{dataset}/answers_{run_name}.jsonl`
(for OmniMedVQA, one file per modality: `answers_{modality}_{run_name}.jsonl`).
`output_root` defaults to `./outputs/eval/` and can be overridden with
`--output-root` or via the `output_root` key in `datasets.yaml`.

## 4. Scoring

Once predictions are written, score them with the dataset-specific helpers:

```bash
scripts/eval_vqa_rad.sh     baseline
scripts/eval_slake.sh       baseline
scripts/eval_pathvqa.sh     baseline
scripts/eval_omnimed.sh     baseline
scripts/eval_biomedparse.sh baseline
```

Or, equivalently, call the Python scorers directly:

```bash
python run_eval.py \
    --gt   /path/to/VQA-RAD/test.json \
    --pred outputs/eval/VQA-RAD/answers_baseline.jsonl \
    --eval_res outputs/eval/VQA-RAD/eval_baseline.txt

# Multiple-choice (OmniMedVQA, one call per modality):
python run_eval_mc.py \
    --gt   /path/to/OmniMedVQA/.../Modality_CT.json \
    --pred outputs/eval/OmniMedVQA/answers_CT_baseline.jsonl \
    --eval_res outputs/eval/OmniMedVQA/eval_CT_baseline.txt
```

`run_eval.py` reports exact match, F1, precision, recall, BLEU-{1,2,3,4} on
open-ended items and yes/no accuracy on closed-ended items. Pass
`--modality_split` (used by `eval_biomedparse.sh`) to additionally break results
down by the `modality` field of each ground-truth item.

## 5. End-to-end

To run inference + scoring on every supported dataset in one command:

```bash
scripts/eval_all.sh baseline
```

## 6. Prediction file format

Each line of `answers_*.jsonl` is:

```json
{"question_id": "<id>", "prompt": "<question>", "text": "<model output>",
 "gt_ans": "<ground truth>", "metadata": {}}
```

This is also the format consumed by `run_eval.py` and `run_eval_mc.py`, so any
external predictor that follows it can be scored with the same tools.
