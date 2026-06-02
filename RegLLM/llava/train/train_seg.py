# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import torch.nn as nn
import torchvision.transforms as T

import transformers
import tokenizers
import deepspeed
from RegLLM.llava import model
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

# from transformers import Qwen2_5_VLProcessor

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, VQType

# Final model with segmentation
from RegLLM.RegSeg import RegSegForCausalLM
from RegLLM.RegSeg_qwen2 import RegSegForCausalLM as RegSegForCausalLM_qwen2

from PIL import Image

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrained_llm_path: Optional[str] = field(default="")
    peft_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_vision_tuning_embedding: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_vision_vq_type: Optional[str] = field(default='CLIP')
    mm_vision_weight_path: Optional[str] = field(default='/path/to/clip/weights')
    use_regtok: Optional[bool] = field(default=False)
    use_region_tokens: Optional[bool] = field(default=False)
    use_sep_proj: Optional[bool] = field(default=False)
    # config path for reg_tok
    regtok_config_path: Optional[str] = field(default=None)
    regtok_weight_path: Optional[str] = field(default=None)
    output_segmentation: Optional[bool] = field(default=False)
    use_seg_loss: Optional[bool] = field(default=False)
    modality_num: Optional[int] = field(default=0)
    codebook_size: Optional[int] = field(default=32)
    codebook_path: Optional[str] = field(default=None)
    resize_embedding: Optional[bool] = field(default=False)
    train_codebook: Optional[bool] = field(default=False)
    load_codebook_embeddings: Optional[bool] = field(default=False)
    merge_codebook: Optional[bool] = field(default=False)
    align_regtok: Optional[bool] = field(default=False)
    seg_align_stage: Optional[bool] = field(default=False)
    train_all_embeddings: Optional[bool] = field(default=False)
    use_moe: Optional[bool] = field(default=False)
    moe_num_experts: Optional[int] = field(default=4)
    final_tune_stage: Optional[bool] = field(default=False)
    use_lightweight_decoder: Optional[bool] = field(default=True)
    final_freeze_projector: Optional[bool] = field(default=False)
    use_seg_token: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    evaluation_strategy: str = "steps"
    eval_steps: int = 0
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    max_grad_norm: float = field(default=1.0)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    # to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    to_return = {k: v for k, v in to_return.items()}
    return to_return

class SplitEmbedding(nn.Module):
    def __init__(self, base_emb: nn.Embedding, codebook_token_ids: list[int], codebook_emb_tensors=None):
        super().__init__()
        self.base_emb = base_emb
        for p in self.base_emb.parameters():
            p.requires_grad = False

        self.codebook_ids = torch.tensor(codebook_token_ids, dtype=torch.long)
        self.id_to_local = {tid.item(): i for i, tid in enumerate(self.codebook_ids)}
        self.codebook_emb = nn.Embedding(len(codebook_token_ids),
                                         base_emb.embedding_dim)
        if codebook_emb_tensors is not None:
            print("Copying codebook embeddings from codebook_emb_tensors!")
            self.codebook_emb.weight.data.copy_(codebook_emb_tensors)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_out = self.base_emb(input_ids)

        # Prepare idx_map (mapping from global token id → local codebook id)
        idx_map = torch.full_like(input_ids, -1, dtype=torch.long)
        # print(self.codebook_ids, "codebook_ids", input_ids)
        for i, tid in enumerate(self.codebook_ids):
            idx_map[input_ids == tid] = i

        # mask where codebook tokens appear
        mask = idx_map >= 0

        if mask.any():
            # Compute codebook embeddings for masked positions
            code_emb = self.codebook_emb(idx_map.clamp(min=0))
            # Create a new tensor: where mask True → use codebook_emb, else → base_out
            out = torch.where(mask.unsqueeze(-1), code_emb, base_out)
        else:
            out = base_out

        return out

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', "mask_decoder", "region", 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            name_to_Add = names[0] if len(names) == 1 else names[-1]
            if len(name_to_Add) > 1:
                lora_module_names.add(name_to_Add)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str, model_args=None):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if model_args and model_args.train_codebook:
            keys_to_match.append('lm_head')
            if model_args.train_all_embeddings:
                keys_to_match.append('embed_tokens')
            else:
                keys_to_match.append('codebook_emb')
        if model_args and model_args.use_seg_loss:
            keys_to_match.extend(['token_projection', 'mask_decoder'])

        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        print("Using deepspeed to save model...")
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    if isinstance(trainer.model, FSDP):
        print("Saving model state dict (FSDP)...")
        # Gather shards to CPU one layer at a time; only rank 0 receives the full dict.
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            trainer._save(output_dir, state_dict=cpu_state_dict)
        return

    state_dict = trainer.model.state_dict()
    print("Saving model state dict...")
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # print(image_token_index,"image token index")
    for token_name, token_id in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids):
        if "im_start" in token_name:
            im_start = token_id
        elif "im_end" in token_name:
            im_end = token_id
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # print(conv, "conversations")
            encode_id = tokenizer.apply_chat_template(conv)
            # print("Conversation:", conv)
            # print("Encoded IDs:", encode_id)
            # print("Decoded:", tokenizer.decode(encode_id))

            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        # print("Input Target 01:", len(target))
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
        # print("Input Target 02:", len(target))
    # print(IMAGE_TOKEN_INDEX, "image token index", input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    # print(input_ids, "input_ids")
    # print("Decoded:", tokenizer.decode(input_ids[0]))
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print("Preprocessing...", conversation_lib.default_conversation.sep_style)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_size = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # process codes and segmentation files
        mask_files_loaded_list = []
        if self.data_args.output_segmentation and 'mask_orders' in sources[0] and 'mask_files' in sources[0]:
            mask_orders = sources[0]['mask_orders']
            mask_files = sources[0]['mask_files']
            # read mask files
            if mask_files is not None and mask_orders is not None:
                mask_files_loaded = dict()
                for key in mask_files:
                    mask_file = mask_files[key]
                    mask = Image.open(mask_file).convert('L')
                    mask = T.ToTensor()(mask)
                    mask = (mask > 0.5).float()
                    mask_files_loaded[key] = mask
                for mask_id in mask_orders:
                    mask_id = str(mask_id)
                    if mask_id in mask_files_loaded:
                        mask = mask_files_loaded[mask_id]
                        # Do something with the mask    
                        mask_files_loaded_list.append(mask)

        if 'image' in sources[0]:

            image_files = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            # print(image_folder, image_file, "images")
            # if type(image_file) is list:
            #     image_file = image_file[0]
            if not isinstance(image_files, list):
                image_files = [image_files]
            
            # Remove all existing <image> placeholders from all conversation contents
            for conv in sources[0]['conversations']:
                conv['value'] = conv['value'].replace('<image>', '').strip()
            # Insert the correct number of <image> tokens at the beginning of the first conversation
            if len(image_files) > 0:
                image_tokens = '\n'.join(['<image>'] * len(image_files))
                sources[0]['conversations'][0]['value'] = f"{image_tokens}\n{sources[0]['conversations'][0]['value']}"
            
            images = []
            for image_file in image_files:
                # print(os.path.join(image_folder, image_file))
                try:
                    final_path = os.path.join(image_folder, image_file)
                    image = Image.open(final_path)
                    # print(f"Loaded image {final_path}")
                except Exception as e:
                    print(f"Error loading image {image_file}: {e}")
                    image = Image.new('RGB', (224, 224), (0, 0, 0))
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    try:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                        if self.image_size is None:
                            self.image_size = image.shape
                            self._ideal_image = image
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        # image = torch.zeros(3, processor.crop_size['height'], processor.crop_size['width'])
                        image = torch.zeros(self.image_size, dtype=self._ideal_image.dtype, device=self._ideal_image.device)
                images.append(image)
            images = torch.stack(images)  # shape: [num_images, C, H, W]
            # data_dict['images'] = images
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
            
        

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = images
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        if self.data_args.output_segmentation and mask_files_loaded_list:
            data_dict['mask_labels'] = torch.stack(mask_files_loaded_list)
        return data_dict


class LazySupervisedDatasetCambrain(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDatasetCambrain, self).__init__()

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.data_path = data_path
        self.index = []
        self.length_list = []
        self.modality_length_list = []
        
        with open(self.data_path, 'r') as file:
            offset = 0
            for line in file:
                sample = json.loads(line.strip())
                img_tokens = 128 if self._has_image(sample) else 0
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                self.length_list.append(cur_len + img_tokens)
                modality_len = cur_len if 'image' in sample else -cur_len
                self.modality_length_list.append(modality_len)

                self.index.append(offset)
                offset += len(line)


    def _has_image(self, sample: dict) -> bool:
        return "image" in sample and not str(sample['image']) in ['', 'None', 'none', 'nan']

    def __len__(self):
        return len(self.index)

    @property
    def lengths(self):
        return self.length_list

    @property
    def modality_lengths(self):
        return self.modality_length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        with open(self.data_path, 'r') as file:
            file.seek(self.index[i])
            line = file.readline()
            sources = json.loads(line.strip())

        dat = sources
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        has_image = self._has_image(dat)

        if has_image:
            image_file = dat['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except:
                print(f"Error loading image {image_file}")
                return self.__getitem__(0)
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        if (data_dict['labels']!=IGNORE_INDEX).sum()==0:
            return self.__getitem__(0)
        # image exist in the data
        if has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if "mask_labels" in instances[0] and instances[0]['mask_labels'] is not None:
            batch['mask_labels'] = torch.stack([instance['mask_labels'] for instance in instances]) 
            # print(batch['mask_labels'].shape, "batch['mask_labels']")
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
    #                             data_path=data_args.data_path,
    #                             data_args=data_args)
    # from torch.utils.data import random_split
    # eval_dataset = None
    full_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    train_dataset = full_dataset
    # if data_args.eval_steps > 0:
    #     val_size = 10000
    #     train_size = len(full_dataset) - val_size
    #     train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size])
    #     rank0_print(f"Created train/val split: {train_size}/{val_size}")
    # train_dataset = LazySupervisedDatasetCambrain(tokenizer=tokenizer,
    #                         data_path=data_args.data_path,
    #                         data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                data_collator=data_collator)

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = logits.argmax(-1)
#     # Mask out padding/ignore_index
#     mask = labels != IGNORE_INDEX
#     correct = (predictions[mask] == labels[mask]).sum()
#     total = mask.sum()
#     accuracy = (correct / total).item() if total > 0 else 0.0
#     return {"accuracy": accuracy}

import torch.nn.functional as F
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    mask = labels != IGNORE_INDEX
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    accuracy = (correct / total).item() if total > 0 else 0.0

    # Compute loss (cross-entropy)
    # logits: [batch, seq_len, vocab_size], labels: [batch, seq_len]
    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)
    shift_logits = logits_tensor[..., :-1, :].contiguous()
    shift_labels = labels_tensor[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean"
    ).item()

    return {"accuracy": accuracy, "loss": loss}

def train(attn_implementation=None):
    addr = os.environ.get('MASTER_ADDR', None)
    port = os.environ.get('MASTER_PORT', None)
    print('\n', addr, port)

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Translate the string to enum type right after parsing
    print(model_args, training_args)
    print(model_args.mm_vision_vq_type)
    if model_args.mm_vision_vq_type is not None:
        model_args.mm_vision_vq_type = VQType[model_args.mm_vision_vq_type]

    rank0_print(f"Inspecting experiment hyperparameters:\n")
    rank0_print(f"model_args = {vars(model_args)}\n\n")
    rank0_print(f"data_args = {vars(data_args)}\n\n")
    rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    print("Using conversation template:", model_args.version)
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    codebook_token_ids = []
    if model_args.output_segmentation:
        # add special tokens to the tokenizer
        if model_args.use_seg_token:
            codebooks = ["[SEG]"]
        else:
            codebooks = [f"[M{i}_{j}]" for i in range(model_args.modality_num) for j in range(model_args.codebook_size)]
        tokenizer.add_tokens(codebooks)
        codebook_token_ids.extend(tokenizer.convert_tokens_to_ids(codebooks))
    
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **bnb_model_from_pretrained_args,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                if not model_args.output_segmentation:
                    if "qwen2" in model_args.model_name_or_path.lower() or "llava_qwen2" in model_args.model_name_or_path.lower():
                        model = LlavaQwen2ForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            low_cpu_mem_usage=False,
                            **bnb_model_from_pretrained_args,
                        )
                    else:
                        model = LlavaQwenForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            # attn_implementation=attn_implementation,
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            low_cpu_mem_usage=False,
                            **bnb_model_from_pretrained_args,
                        )
                else:
                    print("use RegSegForCausalLM!")
                    regseg_args = {
                        "seg_token_ids": codebook_token_ids,
                        "seg_grid_size": (1, 1) if model_args.use_seg_token else (32, 18),
                        "use_seg_loss": model_args.use_seg_loss,
                        "train_all_embeddings": model_args.train_all_embeddings,
                        "decoder_dim": 1024,
                        "ce_loss_weight": 1.0,
                        "mask_loss_weight": 1.0,
                        "dice_loss_weight": 1.0,
                        "bce_loss_weight": 1.0,
                        "use_sep_proj": model_args.use_sep_proj,
                        "use_lightweight_decoder": model_args.use_lightweight_decoder,
                        "load_codebook_embeddings": model_args.load_codebook_embeddings,
                    }
                    model_path = model_args.model_name_or_path
                    if model_args.pretrained_llm_path is not None and len(model_args.pretrained_llm_path) > 2:
                        model_path = model_args.pretrained_llm_path
                    if "qwen2" in model_args.model_name_or_path.lower() or "llava_qwen2" in model_args.model_name_or_path.lower():
                        model = RegSegForCausalLM_qwen2.from_pretrained(
                            model_path,
                            cache_dir=training_args.cache_dir,
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            low_cpu_mem_usage=False,
                            **regseg_args,
                            **bnb_model_from_pretrained_args
                        )
                    else:
                        model = RegSegForCausalLM.from_pretrained(
                            model_path,
                            cache_dir=training_args.cache_dir,
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            low_cpu_mem_usage=False,
                            **regseg_args,
                            **bnb_model_from_pretrained_args
                        )
                    # Inspect or modify before merging
                    # print("Codebook before merge:", model.get_input_embeddings().codebook_emb.weight[-1])

                    # Merge and prepare for full FT
                    if model_args.merge_codebook:
                        model.merge_codebook_into_base(unfreeze_base=True, remove_split=True)
                    # print("Codebook after merge:", model.get_input_embeddings().base_emb.weight[-1])
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        
    # if training_args.gradient_checkpointing:
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)
    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        if model_args.use_moe:
            from llava.peft import LoraConfig, get_peft_model, PeftModel
        else:
            from peft import LoraConfig, get_peft_model, PeftModel
        if model_args.peft_path and os.path.exists(model_args.peft_path):
            rank0_print(f"Loading LoRA adapters from: {model_args.peft_path}")
            # dtype placement before wrapping
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                elif training_args.fp16:
                    model.to(torch.float16)
            # Set is_trainable=True to keep training LoRA; set to False to freeze for inference/further codebook-only finetuning
            model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=True)
        else:
            target_modules=find_all_linear_names(model)
            print(target_modules)
            if model_args.use_moe:
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=training_args.lora_dropout,
                    bias=training_args.lora_bias,
                    task_type="CAUSAL_LM",
                    lora_nums=model_args.moe_num_experts
                )
            else:
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=training_args.lora_dropout,
                    bias=training_args.lora_bias,
                    task_type="CAUSAL_LM",
                )

            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)



    if model_args.vision_tower is not None:
        print("load vision tower!")
        # print(model)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        # print(model)
        vision_tower = model.get_vision_tower()
        # print(vision_tower, "vision tower")
        vision_tower = vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        print(vision_tower.dtype, "dbwe")
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if model_args.use_region_tokens and model_args.use_sep_proj:
                for p in model.get_model().region_mm_projector.parameters():
                    p.requires_grad = True

        model.config.mm_vision_tuning_embedding = training_args.mm_vision_tuning_embedding = model_args.mm_vision_tuning_embedding
        if model_args.mm_vision_tuning_embedding:
            for p in model.get_model().vision_tower.learnable_codebook_embedding.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            if model.get_model().region_mm_projector is not None:
                for p in model.get_model().region_mm_projector.parameters():
                    p.requires_grad = False

        
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            if model.get_model().region_mm_projector is not None:
                model.get_model().region_mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        
        vq_type = getattr(model_args, 'mm_vision_vq_type')
        model.config.mm_vision_vq_type = str(vq_type.name)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            print('visual tokenizer specifications:')
            
            embed_dim = vision_tower.hidden_size
            num_codebook_tokens = getattr(vision_tower, 'num_codebook_tokens', 0)
            img_size = vision_tower.config.image_size
            num_patches_per_side = vision_tower.num_patches_per_side
            print('------type: %s-------' % str(vq_type))
            print('codebook numxdim: %dx%d' % (num_codebook_tokens, embed_dim))
            print('img_size: %d, patch num per side: %d, compression: %d' % (img_size, num_patches_per_side, img_size//num_patches_per_side))

    if model_args.output_segmentation and not model_args.load_codebook_embeddings:
        model.resize_token_embeddings(len(tokenizer))
        print("Current vocab size & model embedding size: ", len(tokenizer), model.get_input_embeddings().weight.size())
    elif model_args.train_codebook and not model_args.merge_codebook:
        print("Current vocab size & model embedding size that is already split: ", len(tokenizer), model.get_input_embeddings().base_emb.weight.size(), model.get_input_embeddings().codebook_emb.weight.size())
    ## load the projected codebook
    codebook_path = model_args.codebook_path
    codebook_vectors, mapped = None, None
    if codebook_path:
        projected_codebook = torch.load(codebook_path)
        try:
            if len(codebook_token_ids) > 0:
                input_emb = model.get_input_embeddings().weight      # (vocab_size, emb_dim)
                emb_dtype = input_emb.dtype

                # try to get codebook embeddings from vision tower
                codebook_vectors = projected_codebook
                # fallback: no codebook vectors available -> skip initialization
                if codebook_vectors is None:
                    print("Warning: no vision codebook vectors found; skipping codebook token init.")
                else:
                    # Move codebook vectors to embedding device/dtype
                    device = input_emb.device
                    codebook_vectors = codebook_vectors.to(device)

                    # apply projector if available
                    mapped = None
                    mm_proj = None
                    if hasattr(model, "get_model") and hasattr(model.get_model(), "mm_projector"):
                        if not model_args.use_sep_proj:
                            mm_proj = model.get_model().mm_projector
                        else:
                            if hasattr(model.get_model(), "region_mm_projector"):
                                mm_proj = model.get_model().region_mm_projector
                                print("use region_mm_projector for codebook", mm_proj)

                    # Some mm_projector implementations expect input shape (batch, code_dim)
                    # convert the types of codebook vectors
                    codebook_vectors = codebook_vectors.to(mm_proj[0].weight.dtype if mm_proj else emb_dtype)
                    if mm_proj is not None:
                        mm_proj = mm_proj.to(device)
                        with torch.no_grad():
                            mapped = mm_proj(codebook_vectors)
                    # finally assign mapped vectors into the new token slots
                    with torch.no_grad():
                        mapped = mapped.to(emb_dtype)
                        n_code = mapped.size(0)
                        for idx, token_id in enumerate(codebook_token_ids):
                            # if more token ids than code vectors, wrap around
                            src_idx = idx % n_code
                            input_emb[token_id].copy_(mapped[src_idx])
                    print(f"Initialized {len(codebook_token_ids)} codebook token embeddings.")
        except Exception as e:
            print("Exception while initializing codebook tokens:", e)
        if model_args.model_name_or_path == "Qwen/Qwen3-8B" and not model_args.seg_align_stage:
            print("Merge codebook into base for Qwen/Qwen3-8B")
            model.merge_codebook_into_base(unfreeze_base=True, remove_split=True)
    if model_args.align_regtok:
        # make the mask_decoder trainable
        for name, param in model.named_parameters():
            if "mask_decoder" in name or "token_projection" in name:
                param.requires_grad = True
    if model_args.use_sep_proj:
        for name, param in model.named_parameters():
            if "region_mm_projector" in name:
                print("use region_mm_projector ")
                print(name)
                param.requires_grad = True
            
    if model_args.seg_align_stage:
        for name, param in model.named_parameters():
            if "mask_decoder" in name or "token_projection" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    if model_args.final_tune_stage:
        for name, param in model.named_parameters():
            if "mask_decoder" in name or "token_projection" in name or "mm_projector" in name:
                param.requires_grad = True
        if model_args.use_lightweight_decoder:
            model.base_model.model.model.mask_decoder = model.base_model.model.model.mask_decoder.float()
        if model_args.use_sep_proj:
            print("use region_mm_projector as float32")
            model.base_model.model.model.region_mm_projector = model.base_model.model.model.region_mm_projector.float()
        model.base_model.model.model.token_projection = model.base_model.model.model.token_projection.float()
        if model_args.final_freeze_projector:
            for name, param in model.named_parameters():
                if "mm_projector" in name:
                    param.requires_grad = False
        else:
            model.base_model.model.model.mm_projector = model.base_model.model.model.mm_projector.float()
        model.base_model.model.model.embed_tokens.codebook_emb = model.base_model.model.model.embed_tokens.codebook_emb.float()
    if model_args.train_codebook:
        if model_args.train_all_embeddings:
            trainable_parameters = ['embed_tokens', "lm_head"]
        else:
            trainable_parameters = ['codebook_emb']
            if model_args.final_tune_stage:
                trainable_parameters.append("lm_head")
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in trainable_parameters):
                param.requires_grad = True

        if not model_args.train_all_embeddings and type(model.get_input_embeddings()) is nn.Embedding:
            base_emb = model.get_input_embeddings()
            split_emb = SplitEmbedding(base_emb, codebook_token_ids, mapped)
            model.set_input_embeddings(split_emb)

            print(f"SplitEmbedding created outside: base_emb frozen, "
                f"{len(codebook_token_ids)} codebook tokens trainable.")
    if model_args.peft_path and os.path.exists(model_args.peft_path):
        non_lora_trainables_path = os.path.join(model_args.peft_path, "non_lora_trainables.bin")
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cuda')
        print(f"Loading non-LoRA weights from {non_lora_trainables_path}")
        print(non_lora_trainables.keys())
        missing_keys, unexpected_keys = model.load_state_dict(non_lora_trainables, strict=False)
        print("Unexpected keys:", unexpected_keys)
    

    

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    data_args.eval_steps = training_args.eval_steps
    data_args.output_segmentation = model_args.output_segmentation
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    for (n, p) in model.named_parameters():
        if p.requires_grad:
            rank0_print(f"Training parameter: {n, p.data.dtype}")
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
    # rank0_print(getattr(training_args, 'tune_mm_mlp_adapter', False))
    # rank0_print(model_args.tune_mm_mlp_adapter)
    training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    # rank0_print(getattr(training_args, 'tune_mm_mlp_adapter', False))
    # print("Training model...")
    # print(training_args, "training_args")
    # print(data_module, "train_dataset")
    training_args.do_train=True

    # if training_args.gradient_checkpointing:
    #     # enable non-reentrant GC (LoRA-safe)
    #     print("Enabling non-reentrant gradient checkpointing. Caf")
    #     model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    # compute_metrics=compute_metrics,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir, model_args=model_args)


if __name__ == "__main__":
    train()
