import gc
from typing import Dict, List, Optional, Tuple

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM, LlavaQwen2Model
from source.tokenizer.region_perceiver import LightweightRegionDecoder

from torch.utils.checkpoint import checkpoint


def dice_loss(inputs, targets, num_masks, eps=1e-6):
    inputs = inputs.sigmoid().flatten(2)
    targets = targets.flatten(2)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)

    # Special handling: if both input and target are empty, set Dice=1 (loss=0)
    empty_mask = (targets.sum(-1) == 0)
    # print(empty_mask, targets, inputs, targets.size(), inputs.size())
    dice = (numerator + eps) / (denominator + eps)
    dice[empty_mask] = 1.0

    loss = 1 - dice
    return loss.sum() / (num_masks + 1e-8)
    

def sigmoid_ce_loss(inputs, targets, num_masks, eps=1e-8):
    # Compute BCE normally
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(2).mean(2)  # mean over spatial dims per mask

    # Identify empty masks (no positives)
    empty_mask = (targets.flatten(2).sum(-1) == 0)
    loss[empty_mask] = 0.0

    return loss.sum() / (num_masks + eps)

class RegSegMetaModel:
    def __init__(self, config, **kwargs):
        super(RegSegMetaModel, self).__init__(config)
        self.config = config
        # print(kwargs, "regseg args")
        self._initialize_regseg_modules(**kwargs)

    def _initialize_regseg_modules(
            self,
            seg_token_ids: List[int],
            seg_grid_size: Tuple[int, int] = (32, 18),
            decoder_dim: Optional[int] = 0,
            decoder_layers: int = 2,
            decoder_heads: int = 8,
            ce_loss_weight: float = 1.0,
            mask_loss_weight: float = 1.0,
            dice_loss_weight: float = 1.0,
            bce_loss_weight: float = 1.0,
            eos_coef: float = 0.1,
            loss_type: str = "dice_bce",
            use_seg_loss: bool = False,
            use_lightweight_decoder: bool = True,
        ):
        self.use_seg_loss = use_seg_loss
        self.use_lightweight_decoder = use_lightweight_decoder
        if seg_token_ids is None or len(seg_token_ids) == 0:
            raise ValueError(f"seg_token_ids must contain the 32×18 special token ids.")

        self.seg_grid_size = seg_grid_size
        self.num_seg_tokens = seg_grid_size[0] * seg_grid_size[1]
        if len(seg_token_ids) != self.num_seg_tokens:
            raise ValueError(f"Expected {self.num_seg_tokens} seg tokens, got {len(seg_token_ids)}.")
        self.decoder_dim = decoder_dim or self.config.hidden_size
        
        self.seg_token_ids_tensor = torch.tensor(seg_token_ids, dtype=torch.long)
        self.seg_token_id_to_order = {int(tid): idx for idx, tid in enumerate(seg_token_ids)}

        if self.use_seg_loss:
            self.token_projection = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.decoder_dim),
            )
            if self.use_lightweight_decoder:
                self.mask_decoder = LightweightRegionDecoder(
                    dim=self.decoder_dim,
                    code_dim=self.decoder_dim,
                    num_layers=decoder_layers,
                    use_multiscale=True,
                    num_classes=None,
                    num_heads=decoder_heads,
                    max_queries=self.num_seg_tokens,
                )

                self.visual_feature_norm = nn.LayerNorm(self.decoder_dim)
        else:
            self.token_projection, self.mask_decoder = None, None

        self.ce_loss_weight = ce_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.eos_coef = eos_coef
        self.loss_type = loss_type

    @property
    def seg_token_ids(self) -> torch.Tensor:
        return self.seg_token_ids_tensor


class RegSegModel(RegSegMetaModel, LlavaQwen2Model):
    def __init__(self, config, **kwargs):
        super(RegSegModel, self).__init__(config, **kwargs)
        self.config.use_cache = False
        self.config.mm_use_im_patch_token = False

    def collect_seg_token_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Collect seg token embeddings from hidden_states while accounting for a possible
        image-token prefix in hidden_states that is not present in input_ids.

        hidden_states.shape[1] = hidden_seq_len (image tokens + text tokens)
        input_ids.shape[1] = input_seq_len (only text tokens, image prefix replaced by a single special token)
        We compute offset = hidden_seq_len - input_seq_len and map input token positions
        to hidden_states positions by adding the offset.
        """
        batch_size, hidden_seq_len, _ = hidden_states.shape
        input_seq_len = input_ids.shape[1]

        if hidden_seq_len < input_seq_len:
            raise ValueError(
                f"hidden_states length ({hidden_seq_len}) is shorter than input_ids length ({input_seq_len})."
            )

        offset = hidden_seq_len - input_seq_len

        seg_embeddings = []
        seg_token_set = set(int(t.item()) for t in self.seg_token_ids_tensor)
        for b in range(batch_size):
            token_indices = []
            token_ids = []
            for idx in range(input_seq_len):
                token_id = int(input_ids[b, idx].item())
                if token_id in seg_token_set:
                    # map input index to corresponding index in hidden_states
                    token_indices.append(idx + offset)
                    token_ids.append(token_id)
                    # if len(token_ids) > 5:
                    #     break
            token_indices_t = torch.tensor(token_indices, device=hidden_states.device)
            if len(token_indices) > 0:
                # print(token_indices, "token_indices")
                gathered = hidden_states[b, token_indices_t]
                seg_embeddings.append(gathered)
            else:
                seg_embeddings.append(hidden_states[b, 0:1])

            # if len(token_indices) > 0:
            #     token_indices_t = torch.tensor(token_indices, device=hidden_states.device)
            #     gathered = hidden_states[b, token_indices_t]
            #     seg_embeddings.append(gathered)
            # else:
            #     # No seg tokens: use a zero, detached placeholder to avoid LM gradient updates
            #     pad = torch.zeros_like(hidden_states[b, 0:1])
            #     seg_embeddings.append(pad.detach())


        return torch.stack(seg_embeddings, dim=0)  # (B, N, hidden)

    def decode_masks(
        self,
        region_codes: torch.Tensor,
        image_features=None,
        # use_lightweight_decoder: bool = True,
    ):  
        # region_latents, fused_features = self.mask_decoder(region_codes, image_features=image_features[0], multi_scale_image_features=image_features)
        # if self.use_lightweight_decoder:
        #     print("Using lightweight decoder, ss")
        #     region_latents, fused_features = self.mask_decoder(region_codes, image_features=image_features[0], multi_scale_image_features=image_features, use_mask=True)
        #     seg_logits, class_logits, _, _ = self.mask_decoder.segmentation_cls_head_quant(
        #         region_latents,
        #         fused_features,
        #         output_mask=False,
        #     )
        # else:

        vision_tower = self.get_vision_tower()
        seg_logits, class_logits = vision_tower.vision_tower.region_perceiver.decode_mask(region_codes)    

        return seg_logits, class_logits

class SplitEmbeddingOri(nn.Module):
    def __init__(self, base_emb: nn.Embedding, codebook_token_ids: list[int]):
        super().__init__()
        self.base_emb = base_emb
        for p in self.base_emb.parameters():
            p.requires_grad = False

        self.codebook_ids = torch.tensor(codebook_token_ids, dtype=torch.long)
        self.id_to_local = {tid.item(): i for i, tid in enumerate(self.codebook_ids)}
        self.codebook_emb = nn.Embedding(len(codebook_token_ids),
                                         base_emb.embedding_dim)

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

class SplitEmbedding(nn.Module):
    def __init__(self, base_emb: nn.Embedding, codebook_token_ids: list[int]):
        super().__init__()
        self.base_emb = base_emb
        for p in self.base_emb.parameters():
            p.requires_grad = False

        self.codebook_ids = torch.tensor(codebook_token_ids, dtype=torch.long)
        self.id_to_local = {tid.item(): i for i, tid in enumerate(self.codebook_ids)}
        self.codebook_emb = nn.Embedding(len(codebook_token_ids),
                                         base_emb.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_out = self.base_emb(input_ids)
        return base_out

    @torch.no_grad()
    def merge_into_base(self, unfreeze_base: bool = True) -> nn.Embedding:
        """
        Copy trained codebook_emb rows into base_emb at positions given by codebook_ids.
        Returns the updated base_emb (now containing codebook weights).
        Call this after loading checkpoint but before training.
        """
        # Handle DeepSpeed ZeRO-3 parameter gathering
        try:
            from deepspeed import zero as ds_zero
            gather_base = ds_zero.GatheredParameters(list(self.base_emb.parameters()), modifier_rank=None)
            gather_code = ds_zero.GatheredParameters(list(self.codebook_emb.parameters()), modifier_rank=None)
        except Exception:
            # Fallback for non-DS environments
            class _Noop:
                def __enter__(self): return self
                def __exit__(self, *args): return False
            gather_base = _Noop()
            gather_code = _Noop()

        with gather_base, gather_code:
            base_w = self.base_emb.weight
            code_w = self.codebook_emb.weight
            
            # Type and device alignment
            if base_w.dtype != code_w.dtype:
                code_w = code_w.to(base_w.dtype)
            code_ids = self.codebook_ids.to(base_w.device)
            
            # Sanity check
            if code_ids.max().item() >= base_w.shape[0]:
                raise ValueError(
                    f"Codebook id {code_ids.max().item()} >= base vocab size {base_w.shape[0]}. "
                    "Did you resize_token_embeddings?"
                )
            
            # Copy codebook rows into base embedding
            base_w.index_copy_(0, code_ids, code_w.to(base_w.device))
            print(f"Merged {len(code_ids)} codebook embeddings into base at indices {code_ids[:5].tolist()}...")

        # Optionally unfreeze for full finetuning
        for p in self.base_emb.parameters():
            p.requires_grad = unfreeze_base
        
        if unfreeze_base:
            print("Base embedding unfrozen for full finetuning.")
        
        return self.base_emb

class RegSegForCausalLM(LlavaQwen2ForCausalLM):
    def __init__(self, config, **kwargs):
        seg_token_ids = kwargs.pop("seg_token_ids", None)
        self.use_seg_loss = kwargs.get("use_seg_loss", False)
        self.codebook_token_ids = seg_token_ids
        self.train_all_embeddings = kwargs.pop("train_all_embeddings", False)
        self.load_codebook_embeddings = kwargs.pop("load_codebook_embeddings", False)
        if not hasattr(config, "use_sep_proj"):
            config.use_sep_proj = kwargs.pop("use_sep_proj", False) 
        print("Initializing RegSegForCausalLM with config:", config)
        super().__init__(config)
        self.model = RegSegModel(config, seg_token_ids=seg_token_ids, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # if not self.train_all_embeddings and self.load_codebook_embeddings:
        if self.load_codebook_embeddings:
            self._init_codebook()
    
    def _init_codebook(self):
        # Replace the embedding layer with SplitEmbedding
        base_emb = self.get_input_embeddings()
        if self.train_all_embeddings:
            split_emb = SplitEmbedding(base_emb, self.codebook_token_ids)
        else:
            split_emb = SplitEmbeddingOri(base_emb, self.codebook_token_ids)
        self.set_input_embeddings(split_emb)

        print(f"SplitEmbedding created: base_emb {base_emb.weight.shape} frozen, "
            f"{len(self.codebook_token_ids)} codebook tokens trainable.")
    
    @torch.no_grad()
    def merge_codebook_into_base(self, unfreeze_base: bool = True, remove_split: bool = True):
        """
        Merge trained codebook_emb into base_emb and optionally remove SplitEmbedding.
        Call this AFTER from_pretrained has loaded checkpoint weights.
        
        Args:
            unfreeze_base: If True, make base_emb trainable for full finetuning
            remove_split: If True, replace SplitEmbedding with plain base_emb
        """
        emb = self.get_input_embeddings()
        
        if not isinstance(emb, SplitEmbedding):
            print("No SplitEmbedding found; nothing to merge.")
            return
        
        # Merge codebook weights into base
        base = emb.merge_into_base(unfreeze_base=unfreeze_base)
        
        if remove_split:
            # Replace input embeddings with the unified base embedding
            self.set_input_embeddings(base)
            print("Removed SplitEmbedding wrapper; using unified base embedding.")
        else:
            print("Kept SplitEmbedding wrapper (merged weights remain in base_emb).")


    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mask_labels: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        del kwargs['num_items_in_batch']
        if "output_hidden_states" in kwargs:
            kwargs.pop("output_hidden_states")
        language_output = super().forward(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_multi_scale_features=False,
            **kwargs,
        )
        last_hidden_state = language_output.hidden_states[-1]

        region_codes = None
        if self.use_seg_loss:
            seg_embeddings = self.model.collect_seg_token_embeddings(last_hidden_state, input_ids)
            if self.model.token_projection[0].weight.dtype == torch.float32:
                # seg_embeddings = seg_embeddings.to(torch.float32)
                seg_embeddings = seg_embeddings.to(torch.float32)
            region_codes = self.model.token_projection(
                seg_embeddings.view(-1, seg_embeddings.size(-1))
            ).view(seg_embeddings.shape[0], seg_embeddings.shape[1], -1)
        # print("Region codes collected.", region_codes.shape)
        seg_logits, class_logits = None, None

        if region_codes is not None:
            seg_logits, class_logits = self.model.decode_masks(region_codes)

        ce_loss = language_output.loss
        mask_loss = None
        dice_loss_val = None
        bce_loss_val = None
        if seg_logits is not None:
            B, N, Hs, Ws = seg_logits.shape
            # print("seg_logits's N", N)
            device = seg_logits.device
            dtype = seg_logits.dtype
            # print(mask_labels, "mask_labels")
            if mask_labels is None:
                target_masks = torch.zeros((B, N, Hs, Ws), device=device, dtype=dtype)
            else:
                # mask_labels is expected as list of per-sample tensors (n_i,1,H,W) or (n_i,H,W)
                target_masks = torch.zeros((B, N, Hs, Ws), device=device, dtype=dtype)
                for b_idx in range(B):
                    lab = mask_labels[b_idx].to(device)
                    # normalize dims to (n_lab, H, W)
                    if lab.dim() == 4 and lab.shape[1] == 1:
                        lab = lab.squeeze(1)
                    if lab.dim() != 3:
                        # unexpected format: skip (leave zeros)
                        continue
                    n_lab, tm_h, tm_w = lab.shape
                    if (tm_h, tm_w) != (Hs, Ws):
                        # interpolate lab to (Hs, Ws) in one vectorized op
                        lab = F.interpolate(lab.unsqueeze(1), size=(Hs, Ws), mode="bilinear", align_corners=False).squeeze(1)
                    copy_n = min(n_lab, N)
                    if copy_n > 0:
                        target_masks[b_idx, :copy_n] = lab[:copy_n].to(dtype)

            num_masks = float(B * N)
            target_masks = target_masks
            bce_loss_val = sigmoid_ce_loss(seg_logits, target_masks, num_masks)
            dice_loss_val = dice_loss(seg_logits, target_masks, num_masks)
            mask_loss = self.model.mask_loss_weight * (
                self.model.dice_loss_weight * dice_loss_val + self.model.bce_loss_weight * bce_loss_val
            )
            # print(f"mask_loss: {mask_loss.item()}")

        total_loss = None
        if ce_loss is not None:
            ce_loss = ce_loss * self.model.ce_loss_weight
            total_loss = ce_loss
        # print("mask loss:", mask_loss, ce_loss)
        if mask_loss is not None:
            # print(f"mask_loss: {mask_loss.item()}, ce_loss: {ce_loss.item()}")
            total_loss = mask_loss if total_loss is None else total_loss + mask_loss

        if hasattr(language_output, "to_dict"):
            lang_dict = language_output.to_dict()
        else:
            lang_dict = dict(language_output)

        result = dict(lang_dict)
        result.update(
            {
                "loss": total_loss,
                "ce_loss": ce_loss,
                "mask_loss": mask_loss if mask_loss is not None else torch.tensor(0.0),
                "dice_loss": dice_loss_val if dice_loss_val is not None else torch.tensor(0.0),
                "bce_loss": bce_loss_val if bce_loss_val is not None else torch.tensor(0.0),
                # "seg_logits": seg_logits,
                # "class_logits": class_logits,
            }
        )

        # del last_hidden_state, seg_embeddings, region_codes
        # del image_features, seg_logits, target_masks
        # if multi_scale_image_features is not None:
        #     del multi_scale_image_features
        
        # # Force cleanup every N steps to prevent fragmentation
        # if torch.cuda.is_available() and self.training:
        #     step = getattr(self, '_step_counter', 0)
        #     self._step_counter = step + 1
        #     if step % 20 == 0:  # adjust frequency
        #         gc.collect()
        #         torch.cuda.empty_cache()

        # Safe deletions (not returned in result)
        del last_hidden_state
        if seg_embeddings is not None:
            del seg_embeddings
        if region_codes is not None:
            del region_codes
        if target_masks is not None:
            del target_masks
        if seg_logits is not None:
            del seg_logits
        if class_logits is not None:
            del class_logits
        
        # Force cleanup every N steps to prevent fragmentation
        if torch.cuda.is_available() and self.training:
            step = getattr(self, '_step_counter', 0)
            self._step_counter = step + 1
            if step % 50 == 0:  # 50 is better than 20 (less overhead)
                gc.collect()
                torch.cuda.empty_cache()

        return ModelOutput(result)

    @torch.no_grad()
    def evaluate(
        self,
        images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.generate(
            input_ids,
            images=images.to(torch.bfloat16),
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        generated_ids = outputs.sequences
        hidden_states = outputs.hidden_states[-1]
        seg_embeddings = self.model.collect_seg_token_embeddings(hidden_states, generated_ids)
        region_codes = self.model.token_projection(
            seg_embeddings.view(-1, seg_embeddings.size(-1))
        ).view(seg_embeddings.shape[0], seg_embeddings.shape[1], -1)
        image_features = self.model.extract_visual_features(images.to(hidden_states.dtype))
        seg_logits, class_logits = self.model.decode_masks(region_codes, image_features)
        return ModelOutput(
            {
                "sequences": generated_ids,
                "seg_logits": seg_logits,
                "class_logits": class_logits,
            }
        )



