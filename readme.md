# MedSIGHT: Towards Grounded Visual Comprehension in Medical Large Vision-Language Models

This repository is the official codebase for the ICML 2026 paper
**"MedSIGHT: Towards Grounded Visual Comprehension in Medical Large Vision-Language Models."**

MedSIGHT is a unified medical large vision-language model (Med-LVLM) that performs
medical **visual comprehension, region grounding, and pixel-level segmentation within a
single generative framework**. Unlike prior unified models that bolt on an external
segmentation network and represent every region with a single `[SEG]` token, MedSIGHT
introduces two complementary modules:

- **Region Perceiver** — produces fine-grained, region-centric visual tokens that encode
  spatial detail directly into the LLM's input space, going beyond the patch-level features
  of standard CLIP encoders.
- **Modality-aware Region Codebook** — a set of discrete region codes appended to the LLM
  vocabulary. The LLM emits these codes as symbolic representations of anatomical and
  pathological regions, and they are decoded back through the Region Perceiver to
  reconstruct segmentation masks — enabling end-to-end spatial grounding without a separate
  segmentation model.

Trained on only **72K multimodal instruction pairs**, MedSIGHT achieves state-of-the-art
results across diverse imaging modalities on both medical comprehension and segmentation
benchmarks.

<p align="center">
  <em>Built on Qwen3-8B + UniMed-CLIP (ViT-L/14).</em>
</p>

---

## Highlights

- **Unified comprehension + grounding + segmentation** in one autoregressive model.
- **Region Perceiver**: dual cross-attention between learnable region queries and
  progressively upsampled image features, capturing both global semantics and pixel-level
  detail.
- **Modality-aware codebook**: each of the 18 imaging modalities owns a dedicated set of 32
  discrete region codes (`[C{modality}_{idx}]`), letting the model express *multiple*,
  semantically distinct regions per response.
- **Progressive training pipeline** that stably aligns the Region Perceiver, codebook, and
  LLM stage by stage.
- **DiagSeg** — a new benchmark for *Grounded Diagnostic Segmentation* that requires the
  model to first diagnose, then segment, mirroring real clinical workflows.

---

## Method overview

MedSIGHT takes a medical image `I` and a text prompt `T`. The image is encoded by a frozen
CLIP encoder into patch features `I`, refined by the **Region Perceiver** `R` into region
embeddings `Q_r`. The concatenation `[I; Q_r]` is mapped into the LLM space by a
vision-to-text projector `P(v→t)`. The LLM vocabulary is extended with the **region
codebook** `C`. The LLM jointly reasons over visual embeddings, text, and codebook tokens,
producing text answers that may contain region codes; the hidden states of those codes are
projected back via `P(t→v)` and decoded by the Region Perceiver's segmentation head into
masks.

The model is trained with a **progressive, multi-stage pipeline** (Algorithm 1 in the
paper):

| Stage | What is trained | Data | Code / scripts |
|-------|-----------------|------|----------------|
| 1. Region Perceiver pre-training | Region Perceiver `R` (seg + cls heads) | BiomedParse `D_seg` | `source/` — `train_full.sh` |
| 2. Codebook learning | Modality-aware codebook `C`, `W_q`, `W_m` (`R` frozen) | BiomedParse `D_seg` | `source/` — `train_vq_full.sh` |
| 3. Vision→Text alignment | Vision-to-text projector `P(v→t)` (LLM, encoder, `R` frozen) | PubMedVision `D_v→t` (647K) | `RegLLM/scripts/align_stage1.sh` |
| 4. Text→Vision / codebook alignment | `P(t→v)` + codebook embeddings | grounding set `D_t→v` (60K) | `RegLLM/scripts/align_regtok_1110.sh` |
| 5. Unified grounded instruction tuning | LLM `M` + projectors + codebook | `D_r_inst` (60K) ∪ `D_g_inst` (12K) | `RegLLM/scripts/instruct_region_1110.sh` (full FT) |

Stages 1–2 live under `source/` (Region Perceiver + codebook). Stages 3–5 live under
`RegLLM/` (the LVLM).

---

## Repository layout

```
.
├── source/                       # Region Perceiver + codebook (Stages 1–2)
│   ├── regtok/                   # Region Perceiver & codebook implementation
│   │   ├── region_perceiver.py   #   dual cross-attention region perceiver
│   │   ├── vq_model.py           #   RegTok wrapper (CLIP encoder + perceiver + quantizer)
│   │   ├── norm_ema_quantizer.py #   vector-quantization codebook
│   │   ├── vq_loss.py            #   segmentation / VQ losses
│   │   ├── vq_train.py           #   Region Perceiver / codebook training entry point
│   │   ├── open_clip/, clip/     #   UniMed-CLIP encoder
│   │   └── regtok_config.yaml    #   Region Perceiver / codebook config
│   ├── dataset/                  # BiomedParse segmentation data loaders
│   ├── scripts/                  # Region Perceiver + codebook training scripts
│   ├── utils/                    # training utilities (ddp, ema, logging, ...)
│   └── evaluate_segmentation.py  # standalone Region Perceiver segmentation eval
│
└── RegLLM/                       # MedSIGHT LVLM (Stages 3–5)
    ├── RegSeg.py                 # MedSIGHT model (Qwen3-8B + RegTok + segmentation head)
    ├── RegSeg_qwen2.py           # Qwen2.5-7B variant
    ├── llava/
    │   ├── model/                # model builders + multimodal encoder (RegTok vision tower)
    │   │   └── language_model/   # llava_qwen3.py, llava_qwen2.py, ...
    │   ├── train/                # training entry points (train_mem.py, train_seg.py)
    │   └── eval/                 # evaluation & inference (see RegLLM/llava/eval/README.md)
    │       ├── chatbot.py        #   RegLLMChatbot inference API
    │       ├── inference.py      #   YAML-driven batch inference
    │       ├── run_eval.py       #   open-ended VQA scoring
    │       ├── run_eval_mc.py    #   multiple-choice scoring
    │       ├── configs/          #   model.yaml + datasets.yaml
    │       ├── scripts/          #   inference + per-dataset eval wrappers
    │       └── examples/         #   VQA + segmentation demo notebooks
    └── scripts/                  # MedSIGHT training scripts (alignment + instruction tuning)
```

---

## Environment

```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

pip install qwen-vl-utils==0.0.14

pip install peft==0.18.1 deepspeed==0.18.8 datasets==4.8.4 accelerate==1.12.0 \
    huggingface_hub==0.36.2 scikit-learn==1.7.2 scipy==1.16.3 timm==1.0.13 \
    transformers==4.52.4 ftfy

pip install einops_exts==0.0.4 ftfy==6.3.1 wandb==0.25.1
```

All experiments in the paper were run on 4×H100 GPUs.

---

## Models and data

MedSIGHT relies on three pretrained components plus several datasets:

- **Base LLM** — [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
- **Vision encoder** — [UniMed-CLIP (ViT-L/14)](https://arxiv.org/abs/2412.10372).
- **Region Perceiver + codebook** — pretrained by Stages 1–2 of this repo (or use a released
  checkpoint).

Datasets used:

- **BiomedParse** (`D_seg`) — segmentation/detection corpus for Region Perceiver and codebook
  pre-training.
- **PubMedVision** (`D_v→t`, `D_r_inst`) — 647K image–text pairs for vision-to-text alignment;
  60K sampled for instruction tuning.
- **Grounding instruction set** (`D_t→v`, `D_g_inst`) — region-code-annotated samples
  (60K for text-to-vision alignment, 12K curated for grounded instruction tuning).

> Paths to checkpoints and datasets are configured per script (environment variables in the
> training scripts, and `RegLLM/llava/eval/configs/*.yaml` for evaluation). Edit these to
> match your local layout before running.

---

## Pretrained weights

We release the intermediate and final MedSIGHT checkpoints so you can reproduce any stage of
the pipeline or run inference directly with the final model.

### MedSIGHT checkpoints (this work)

| Component | Stage | Description | Download |
|-----------|-------|-------------|----------|
| **Region Perceiver** | 1 | Pretrained Region Perceiver weights (`regtok_weights.pt`), the input to all downstream stages. | _TODO: add link_ |
| **Region Codebook** | 2 | Trained modality-aware codebook (`codebook.pt`), 18 modalities × 32 codes. | _TODO: add link_ |
| **Stage-1 mm-adapter** | 3 | Vision-to-text projector after the alignment stage (`mm_projector.bin`). | _TODO: add link_ |
| **MedSIGHT (Qwen3-8B)** | 5 | **Final unified model** — comprehension + grounding + segmentation. | 

> To run inference with the final model, point `model_dir`, `regtok_weight_path`, and
> `codebook_tokens_path` in `RegLLM/llava/eval/configs/model.yaml` at the downloaded files.

### External dependencies

| Resource | Description | Link |
|----------|-------------|------|
| **UniMed-CLIP (ViT-L/14)** | Medical CLIP vision encoder used as the frozen image backbone. | https://github.com/mbzuai-oryx/UniMed-CLIP |
| **Qwen3-8B** | Base LLM for the default MedSIGHT model. | https://huggingface.co/Qwen/Qwen3-8B |
| **BiomedParse** | Segmentation/detection corpus for Region Perceiver + codebook pre-training. | https://github.com/microsoft/BiomedParse |
| **PubMedVision** | Image–text corpus for alignment and instruction tuning. | https://huggingface.co/datasets/FreedomIntelligence/PubMedVision |

---

## Training

The pipeline is run in order. Stages 1–2 produce the Region Perceiver and codebook; Stages
3–5 train the LVLM.

### Stage 1–2 — Region Perceiver & codebook (`source/`)

```bash
cd source

# 1. Pre-train the Region Perceiver on BiomedParse (segmentation + classification heads).
#    train_abd.sh trains on a single dataset; train_full.sh trains on all BiomedParse data.
bash scripts/train_full.sh

# 2. Train the modality-aware region codebook on the frozen Region Perceiver.
bash scripts/train_vq_full.sh
```

The Region Perceiver uses `L = 3` refinement layers and 20 region query tokens; the codebook
has `K = 18` modalities × `M = 32` codes of dimension `d_c = 64`. These are set in
`source/regtok/regtok_config.yaml`.

### Stage 3–5 — MedSIGHT LVLM (`RegLLM/`)

```bash
cd RegLLM

# 3. Vision-to-text alignment: train only the multimodal projector.
bash scripts/align_stage1.sh

# 4. Codebook / text-to-vision alignment: integrate the codebook into the LLM vocabulary
#    and align region codes to pixel grounding.
bash scripts/align_regtok_1110.sh

# 5. Unified grounded instruction tuning (choose one):
bash scripts/instruct_region_1110.sh        # full fine-tuning

# Additionally, we provide one example downstream LoRA finetuning script using SLAKE dataset:
bash scripts/finetune_lora_slake.sh    # LoRA fine-tuning
```

A Qwen2.5-7B backbone variant of the alignment stage is provided at
`RegLLM/scripts/rebuttal/alignment_region.sh`.

> **Note on `--mm_vision_select_layer`.** This flag selects which UniMed-CLIP block feeds the
> Region Perceiver and **must match between training and inference**. The released scripts use
> `-2`; set the same value in `RegLLM/llava/eval/configs/model.yaml` when evaluating a model
> trained with these scripts.

---

## Evaluation and inference

All evaluation tooling lives in [`RegLLM/llava/eval/`](RegLLM/llava/eval/), documented in
detail in [`RegLLM/llava/eval/README.md`](RegLLM/llava/eval/README.md). In short:

**Programmatic inference**

```python
from llava.eval.chatbot import RegLLMChatbot

bot = RegLLMChatbot.from_config("llava/eval/configs/model.yaml", device="cuda")

# Visual comprehension
answers, _ = bot.inference("What modality is used to take this image?", "image.jpg")

# Grounded segmentation (requires output_segmentation: true in model.yaml)
result = bot.inference("Please segment the kidney.", "image.jpg", output_seg=True)
print(result["answers"][0])      # text answer with region codes
print(result["mask_logits"].shape)  # decoded mask logits (B, K, H, W)
```

See the demo notebooks in [`RegLLM/llava/eval/examples/`](RegLLM/llava/eval/examples/) for a
VQA walkthrough and a segmentation walkthrough.

**Batch inference + scoring**

```bash
cd RegLLM/llava/eval

# Run inference over a configured dataset (or `all`)
python -m llava.eval.inference \
    --model-config   configs/model.yaml \
    --dataset-config configs/datasets.yaml \
    --dataset        VQA-RAD \
    --run-name       medsight

# Score the predictions
python run_eval.py --gt <test.json> --pred outputs/eval/VQA-RAD/answers_medsight.jsonl \
    --eval_res outputs/eval/VQA-RAD/eval_medsight.txt
```

The shipped configs cover the five comprehension datasets (VQA-RAD, SLAKE, PathVQA,
OmniMedVQA, BiomedParse). OmniMedVQA is multiple-choice and scored with `run_eval_mc.py`.

---

## DiagSeg benchmark

We introduce **DiagSeg** (Grounded Diagnostic Segmentation), a benchmark that evaluates joint
diagnostic reasoning and pixel-level grounding. Given an image and a diagnostic question, a
model must first infer the pathological concept and then produce a segmentation mask aligned
with that diagnosis (e.g., *"What is the abnormality observed? Provide the diagnosis and then
segment it."*). DiagSeg contains 1,655 VQA pairs, each paired with a pixel-level mask, drawn
from the test splits of multiple public segmentation datasets (BiomedParse subset) across
eight modalities (CT, MRI, X-ray, Pathology, Ultrasound, Endoscopy, Dermatoscopy, OCT) and
validated by licensed physicians.

**Download:** _TODO: add DiagSeg link_

---

## Results (summary)

- **Medical visual comprehension** (VQA-RAD, SLAKE, PathVQA, MMMU-Med, OmniMedVQA, DiagSeg
  diagnosis): average 62.3, surpassing the best comprehension-only baseline despite using
  ~9× less instruction-tuning data.
- **Grounded Diagnostic Segmentation (DiagSeg)**: mean Dice 69.9 and diagnosis Recall 58.9,
  outperforming both unified and segmentation-specialist baselines across all eight
  modalities.
- **Text-prompted segmentation (MeCoVQA-G)**: best average Dice (42.8) with strong
  cross-modality generalization.

See the paper for full tables, ablations (region embeddings, codebook integration, training
stages, codebook size), the 7B backbone fairness study, and bounding-box grounding results.

---

## Citation

If you find MedSIGHT useful, please cite:

```bibtex
@inproceedings{chang2026medsight,
  title     = {MedSIGHT: Towards Grounded Visual Comprehension in Medical Large Vision-Language Models},
  author    = {Chang, Aofei and Huang, Le and Boyd, Alex James and Bhatia, Parminder and Kass-Hout, Taha and Ma, Fenglong and Xiao, Cao},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

MedSIGHT builds on [Qwen](https://github.com/QwenLM/Qwen3),
[LLaVA](https://github.com/haotian-liu/LLaVA),
[UniMed-CLIP](https://github.com/mbzuai-oryx/UniMed-CLIP),
[BiomedParse](https://github.com/microsoft/BiomedParse), and the DETR/Mask2Former line of
segmentation work. We thank the authors of these projects for releasing their code and data.
