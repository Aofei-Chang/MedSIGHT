"""MedSIGHT inference chatbot.

`RegLLMChatbot` wraps the Qwen3-based LLaVA model used by MedSIGHT and exposes
two entry points:

  * `inference(text, images, output_seg=False)` for single-turn VQA / segmentation
  * `chat(text, images=None)` for multi-turn streaming chat

Construct it from a YAML config (recommended):

    bot = RegLLMChatbot.from_config("configs/model.yaml")

or directly with a dict of model arguments:

    bot = RegLLMChatbot(model_dir, model_args=model_args, device="cuda")
"""

from __future__ import annotations

import copy
import glob
import json
import os
from threading import Thread
from types import SimpleNamespace
from typing import Optional, Union

import torch
import torch.nn as nn
import yaml
from PIL import Image
from transformers import AutoTokenizer, TextIteratorStreamer

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model import *  # noqa: F401,F403  -- registers Llava models
from llava.model.language_model.llava_qwen3 import LlavaQwenForCausalLM
from RegLLM.RegSeg import RegSegForCausalLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_all_linear_names(model):
    """Collect linear module names eligible for LoRA, excluding multimodal heads."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', "mask_decoder", "region", 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            target = names[0] if len(names) == 1 else names[-1]
            if len(target) > 1:
                lora_module_names.add(target)
    lora_module_names.discard('lm_head')
    return list(lora_module_names)


class SplitEmbedding(nn.Module):
    """Frozen base embedding + trainable embedding for the codebook (region) tokens."""

    def __init__(self, base_emb: nn.Embedding, codebook_token_ids: list[int]):
        super().__init__()
        self.base_emb = base_emb
        for p in self.base_emb.parameters():
            p.requires_grad = False

        self.codebook_ids = torch.tensor(codebook_token_ids, dtype=torch.long)
        self.codebook_emb = nn.Embedding(len(codebook_token_ids), base_emb.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_out = self.base_emb(input_ids)
        idx_map = torch.full_like(input_ids, -1, dtype=torch.long)
        for i, tid in enumerate(self.codebook_ids):
            idx_map[input_ids == tid] = i
        mask = idx_map >= 0
        if mask.any():
            code_emb = self.codebook_emb(idx_map.clamp(min=0))
            return torch.where(mask.unsqueeze(-1), code_emb, base_out)
        return base_out


# ---------------------------------------------------------------------------
# Main chatbot
# ---------------------------------------------------------------------------

class RegLLMChatbot:
    DEFAULT_GENERATION = {
        'do_sample': True,
        'max_new_tokens': 512,
        'min_new_tokens': 1,
        'temperature': 0.2,
        'repetition_penalty': 1.2,
    }

    def __init__(
        self,
        model_dir: str,
        model_args: Optional[Union[dict, SimpleNamespace]] = None,
        device: str = "cuda",
        gen_kwargs: Optional[dict] = None,
    ):
        if model_args is None:
            raise ValueError("model_args must be provided. Use RegLLMChatbot.from_config(...) for the YAML-driven API.")

        self.model_dir = model_dir
        self.peft_path = model_args["peft_path"] if isinstance(model_args, dict) else getattr(model_args, "peft_path", None)
        self.device = device
        self.history: list = []
        self.images: list = []
        self.max_image_num = 6

        self.gen_kwargs = dict(self.DEFAULT_GENERATION)
        if gen_kwargs:
            self.gen_kwargs.update(gen_kwargs)

        if isinstance(model_args, dict):
            model_args = SimpleNamespace(**model_args)
        self.model_args = model_args

        self.init_components()

    # ---- construction from YAML ----------------------------------------
    @classmethod
    def from_config(cls, config_path: str, device: str = "cuda") -> "RegLLMChatbot":
        """Build a chatbot from a YAML config (see configs/model.yaml)."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        gen_kwargs = cfg.pop("generation", None)
        model_dir = cfg.pop("model_dir")

        # The downstream code expects a few legacy field names.
        model_args = dict(cfg)
        model_args.setdefault("pretrained_llm_path", model_dir)
        model_args.setdefault("tokenizer_path", model_dir)
        model_args.setdefault("peft_path", None)

        return cls(model_dir, model_args=model_args, device=device, gen_kwargs=gen_kwargs)

    # ---- model loading -------------------------------------------------
    def init_components(self):
        model_args = self.model_args
        print(f"Loading model from {self.model_dir}")

        codebook_token_ids: list[int] = []
        if model_args.output_segmentation:
            token_ids_path = getattr(model_args, "codebook_tokens_path", None) \
                or os.path.join(self.model_dir, "added_tokens.json")
            if not os.path.exists(token_ids_path):
                raise FileNotFoundError(
                    f"output_segmentation=True but {token_ids_path} is missing. "
                    "Set `codebook_tokens_path` in model.yaml to the added_tokens.json "
                    "of a segmentation-trained checkpoint."
                )
            with open(token_ids_path) as f:
                added = json.load(f)
            codebook_token_ids = [tid for name, tid in added.items() if name.startswith("[")]
            if not codebook_token_ids:
                raise ValueError(
                    f"{token_ids_path} contains no `[...]`-prefixed codebook tokens. "
                    "Point `codebook_tokens_path` at the added_tokens.json from a "
                    "segmentation-trained MedSIGHT checkpoint (which carries the 32×18 "
                    "special tokens), or set output_segmentation: false."
                )
            print(f"Loaded {len(codebook_token_ids)} codebook token ids from {token_ids_path}")

            regseg_args = dict(
                seg_token_ids=codebook_token_ids,
                use_seg_loss=getattr(model_args, "use_seg_loss", True),
                train_all_embeddings=getattr(model_args, "train_all_embeddings", True),
                use_lightweight_decoder=getattr(model_args, "use_lightweight_decoder", False),
                load_codebook_embeddings=getattr(model_args, "load_codebook_embeddings", False),
                use_sep_proj=getattr(model_args, "use_sep_proj", False),
                decoder_dim=1024,
                ce_loss_weight=1.0,
                mask_loss_weight=1.0,
                dice_loss_weight=1.0,
                bce_loss_weight=1.0,
            )
            model = RegSegForCausalLM.from_pretrained(
                model_args.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                **regseg_args,
            )
        else:
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.pretrained_llm_path or model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )

        print("Initialising vision tower")
        model.get_model().initialize_vision_modules(model_args=model_args)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16, device="cuda")

        # ---- tokenizer + weight reload ---------------------------------
        if not model_args.output_segmentation:
            # Plain VQA model: load tokenizer and any safetensors shards from model_dir.
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._configure_tokenizer(tokenizer)

            safetensor_files = sorted(glob.glob(os.path.join(self.model_dir, "*.safetensors")))
            if safetensor_files:
                from safetensors.torch import load_file
                state_dict: dict = {}
                for sf in safetensor_files:
                    state_dict.update(load_file(sf))
                result = model.load_state_dict(state_dict, strict=False)
                print(
                    f"Loaded {len(safetensor_files)} safetensors shards "
                    f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
                )
        else:
            tokenizer_src = getattr(model_args, "tokenizer_path", None) or model_args.pretrained_llm_path
            print(f"Loading tokenizer from {tokenizer_src}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, local_files_only=True)
            self._configure_tokenizer(tokenizer)

        self.tokenizer = tokenizer

        # ---- optional PEFT (LoRA) adapter ------------------------------
        if getattr(model_args, "peft_path", None):
            model = self._load_peft(model, codebook_token_ids)

        vision_tower.to(dtype=torch.bfloat16, device=model.device)
        self.processor = vision_tower.image_processor
        model.eval()
        self.model = model.to(self.device).to(torch.bfloat16)

    def _configure_tokenizer(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
        self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _load_peft(self, model, codebook_token_ids: list[int]):
        model_args = self.model_args
        peft_path = model_args.peft_path

        if getattr(model_args, "resize_embedding", False):
            model.resize_token_embeddings(len(self.tokenizer))

        if (not getattr(model_args, "train_all_embeddings", False)
                and codebook_token_ids
                and isinstance(model.get_input_embeddings(), nn.Embedding)):
            base_emb = model.get_input_embeddings()
            model.set_input_embeddings(SplitEmbedding(base_emb, codebook_token_ids))
            print(f"SplitEmbedding installed ({len(codebook_token_ids)} codebook tokens trainable).")

        print(f"Loading LoRA weights from {peft_path}")
        if getattr(model_args, "use_moe", False):
            from llava.peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=getattr(model_args, "lora_dropout", 0.05),
                bias='none',
                task_type="CAUSAL_LM",
                lora_nums=model_args.lora_nums,
            )
            model = get_peft_model(model, lora_config)
            hlora_weights = torch.load(os.path.join(peft_path, "adapter_model.bin"))
            unexpected = model.load_state_dict(hlora_weights, strict=False)[1]
            if unexpected:
                print(f"Warning: unexpected hlora keys: {unexpected}")
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, peft_path)
            model = model.merge_and_unload()

        non_lora_path = os.path.join(peft_path, "non_lora_trainables.bin")
        if os.path.exists(non_lora_path):
            non_lora = torch.load(non_lora_path, map_location='cuda')
            print(f"Loading non-LoRA weights from {non_lora_path}")
            new_state_dict = {}
            for key, value in non_lora.items():
                new_key = key
                if not getattr(model_args, "use_moe", False):
                    new_key = key.replace("base_model.model", "")
                    if new_key.startswith("."):
                        new_key = new_key[1:]
                new_state_dict[new_key] = value.to("cuda")
            _, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if unexpected:
                print(f"Unexpected non-LoRA keys: {unexpected}")
        return model

    # ---- prompt / image preprocessing ----------------------------------
    def clear_history(self):
        self.images = []
        self.history = []

    def input_moderation(self, t: str) -> str:
        return t.replace('<image>', '')

    def insert_image_placeholder(self, t: str, num_images: int,
                                 placeholder: str = '<image>', sep: str = '\n') -> str:
        for _ in range(num_images):
            t = f"{placeholder}{sep}" + t
        return t

    def get_conv(self, text: str) -> list:
        ret = []
        for human, gpt in self.history or []:
            ret.append({'from': 'human', 'value': human})
            ret.append({'from': 'gpt', 'value': gpt})
        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})
        return ret

    def get_conv_without_history(self, text: str) -> list:
        return [{'from': 'human', 'value': text}]

    def get_image_tensors(self, images):
        processor = self.processor
        tensors = []
        for fp in images:
            if fp is None:
                continue
            if isinstance(fp, str):
                image = Image.open(fp).convert('RGB')
            elif isinstance(fp, Image.Image):
                image = fp
            else:
                raise TypeError(f'Unsupported image type {type(fp)}')
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            tensors.append(image.to(self.device))
        return tensors

    def preprocess_qwen(self, sources, tokenizer, has_image: bool = False,
                        system_message: str = "You are a helpful assistant."):
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")

        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
        tokenizer.chat_template = chat_template

        input_ids = []
        for source in sources:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
            input_id = list(tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}]
            ))
            for conv in source:
                role = roles.get(conv.get("role", conv.get("from")), conv.get("from"))
                content = conv.get("content", conv.get("value"))
                input_id += tokenizer.apply_chat_template([{"role": role, "content": content}])
            for idx, tid in enumerate(input_id):
                if tid == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
            input_ids.append(input_id)
        return torch.tensor(input_ids, dtype=torch.long)

    # ---- single-turn inference -----------------------------------------
    def inference(self, text: str, images=None, output_seg: bool = False):
        """Generate a single-turn response.

        Returns:
          (answers, output_ids)               when output_seg=False
          {"answers": ..., "mask_logits": ...} when output_seg=True
        """
        if images is None:
            images = []
        if isinstance(images, (str, Image.Image)):
            images = [images]

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB')
                valid_images.append(img)
            except Exception:
                print(f'Skipping unreadable image: {img}')
        images = valid_images[:self.max_image_num]

        text = self.input_moderation(text)
        text = self.insert_image_placeholder(text, len(images))

        conv = self.get_conv_without_history(text)
        input_ids = self.preprocess_qwen([conv], tokenizer=self.tokenizer, has_image=True).to(self.device)

        image_tensors = None
        if images:
            list_image_tensors = self.get_image_tensors(images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)

        with torch.inference_mode():
            if output_seg:
                # Greedy decoding gives deterministic segmentation tokens.
                gen_kwargs = dict(self.gen_kwargs, do_sample=False, temperature=1.0, top_p=1.0, top_k=50)
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensors.to(torch.bfloat16),
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    **gen_kwargs,
                )
                output_ids = outputs.sequences
                final_layer_token_states = [step[-1] for step in outputs.hidden_states]
                final_layer_token_states[0] = final_layer_token_states[0][:, -1, :].unsqueeze(1)
                final_layer_token_states = torch.stack(final_layer_token_states, dim=1).squeeze(-2)

                seg_embeddings = self.model.model.collect_seg_token_embeddings(final_layer_token_states, output_ids)
                region_codes = self.model.model.token_projection(
                    seg_embeddings.view(-1, seg_embeddings.size(-1))
                ).view(seg_embeddings.shape[0], seg_embeddings.shape[1], -1)
                seg_logits, _ = self.model.model.decode_masks(region_codes)

                answers = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in output_ids]
                return {"answers": answers, "mask_logits": seg_logits}

            output_ids = self.model.generate(
                input_ids,
                images=image_tensors.to(torch.bfloat16) if image_tensors is not None else None,
                use_cache=True,
                **self.gen_kwargs,
            )
            answers = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in output_ids]
            return answers, output_ids

    # ---- multi-turn streaming chat -------------------------------------
    def chat(self, text: str, images=None) -> str:
        text = self.input_moderation(text)
        if text == '':
            return 'Please type in something'

        if isinstance(images, (str, Image.Image)):
            images = [images]
        if images is None:
            images = []

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB')
                valid_images.append(img)
            except Exception:
                continue
        self.images.extend(valid_images)
        assert len(self.images) < self.max_image_num, f'at most {self.max_image_num} images'

        text = self.insert_image_placeholder(text, len(valid_images))
        conv = self.get_conv(text)
        input_ids = self.preprocess_qwen([conv], tokenizer=self.tokenizer, has_image=True).to(self.device)

        image_tensors = None
        if self.images:
            list_image_tensors = self.get_image_tensors(self.images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensors,
            streamer=streamer,
            use_cache=True,
            **self.gen_kwargs,
        )

        generated_text = ''
        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            for new_text in streamer:
                generated_text += new_text
                print(new_text, end='', flush=True)
            thread.join()

        self.history.append([text, generated_text])
        return generated_text
