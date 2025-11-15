# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/
#   VAR: https://github.com/FoundationVision/VAR

import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
# from .clip import clip
import numpy as np
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
from .norm_ema_quantizer import EmbeddingEMA, l2norm, norm_ema_inplace, kmeans
import torch.distributed as dist
import random

from timm.models.layers import trunc_normal_

#load unimed-CLIP
from .open_clip import create_model_and_transforms, get_mean_std, HFTokenizer
from PIL import Image

# load visual perceiver
from .region_perceiver import RegionPerceiver


def copy_new_embedding(old_embedding, requires_grad=True):
    new_embedding = nn.Embedding(old_embedding.weight.size(0), old_embedding.weight.size(1))
    new_embedding.weight = nn.Parameter(old_embedding.weight.clone())
    new_embedding.weight.requires_grad = requires_grad
    return new_embedding

def drop_scale(original_scales, num_to_drop=1):
    """
    Randomly remove scales from scale list.
    
    Args:
        original_scales: list of scales
        num_to_drop: Number of scales to randomly remove (default 1)
        
    Returns:
        New scale list
    """
    if num_to_drop >= len(original_scales) - 1:
        raise ValueError("Cannot drop that many items")
    
    drop_candidates = list(range(1, len(original_scales)))
    indices_to_drop = set(random.sample(drop_candidates, num_to_drop))
    return [item for i, item in enumerate(original_scales) if i not in indices_to_drop]


@dataclass
class ModelArgs:
    num_queries: int = 16
    codebook_size: int = 32
    codebook_embed_dim: int = 16
    semantic_embed_dim: int = 768
    num_stages: int = 3
    num_stacks: int = 1
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 1.0
    entropy_loss_ratio: float = 1.0
    interpolate_scale_factor: float = 2.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    use_quantization: bool = False
    kmeans: bool = False
    num_classes: int = 0
    infer_interpolate: bool = False
    finetune_codebook_only: bool = False
    use_self_attn: bool = False
    upsample_mode: str = "conv"
    pretrained_weights: str = "/qumulo/shared_data/aofei_summer/CLIPs/unimed_clip_vit_l14.pt"
    # number of modalities (if >0 will enable modality predictor + modality-specific codebooks)
    num_modalities: int = 0
    quant_use_seg: bool = False

def get_model_default_params():
    return dict(img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1152, depth=12, num_heads=12,  
                             mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                             norm_layer='LayerNorm', init_values=0., use_abs_pos_emb=True, 
                             use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


class RegTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ### parameters required by llava ###
        # self.code_dim = 1024
        self.code_dim = config.codebook_embed_dim
        self.num_quries = config.num_queries
        self.hidden_size = 1024

        # self.embed_dim = self.code_dim
        self.embed_dim = config.codebook_embed_dim
        self.n_embed = config.codebook_size
        self.compression = 2**(len(config.encoder_ch_mult) - 1)
        ### load medical image encoder ###
        model_name = 'ViT-L-14-336-quickgelu' # available pretrained weights ['ViT-L-14-336-quickgelu', 'ViT-B-16-quickgelu']
        pretrained_weights = config.pretrained_weights # Path to pretrained weights
        # text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" # available pretrained weights ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"]
        text_encoder_name = None # available pretrained weights ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"]
        mean, std = get_mean_std()
        model, _, preprocess = create_model_and_transforms(
            model_name,
            pretrained_weights,   
            precision='amp',
            force_quick_gelu=True,
            mean=mean, std=std,
            inmem=True,
            text_encoder_name=text_encoder_name,)
        self.image_encoder = model.visual
        self.text_encoder, self.text_tokenizer = None, None
        if text_encoder_name is not None:
            self.text_tokenizer = HFTokenizer(
                text_encoder_name,
                context_length=256,
                **{},)
            self.text_encoder = model.text_encoder.cuda()
        self.unimed_preprocess = preprocess
        self.num_classes=config.num_classes
        self.num_stages=config.num_stages
        self.do_quantize = config.use_quantization

        # quantizer
        self.use_kmeans = config.kmeans
        self.quantizer = None
        if self.do_quantize:
            # self.quantizer = VectorQuantizer(config.codebook_size, self.code_dim, 
            #                     config.commit_loss_beta, config.entropy_loss_ratio,
            #                     config.codebook_l2_norm, config.codebook_show_usage, kmeans=config.kmeans)
            allow_codebook_grad = True if config.finetune_codebook_only else False
            if getattr(config, 'num_modalities', 0) and config.num_modalities > 0:
                # create modality predictor and modality-specific quantizer
                self.modality_predictor = nn.Linear(1024, config.num_modalities)
                # modality classification loss (used when ground-truth modality_label provided during training)
                self.modality_loss_fn = nn.CrossEntropyLoss()
                # instantiate modality-specific quantizer (keys are string indices)
                modality_keys = [str(i) for i in range(config.num_modalities)]
                # quantizer factory is VectorQuantizerST with common parameters
                self.quantizer = ModalitySpecificQuantizer(modality_keys, VectorQuantizerST,
                                                           config.codebook_size, self.code_dim, 0.25,
                                                           entropy_loss_ratio=config.entropy_loss_ratio,
                                                           l2_norm=config.codebook_l2_norm, kmeans=config.kmeans,
                                                           show_usage=config.codebook_show_usage,
                                                           allow_codebook_grad=allow_codebook_grad)
            else:
                self.modality_predictor = None
                self.quantizer = VectorQuantizerST(n_e=config.codebook_size, e_dim=self.code_dim, beta=0.25, entropy_loss_ratio=config.entropy_loss_ratio, l2_norm=config.codebook_l2_norm, kmeans=config.kmeans, show_usage=config.codebook_show_usage, allow_codebook_grad=allow_codebook_grad)

        ### load visual perceiver ###
        self.region_perceiver = RegionPerceiver(
            dim=1024, num_queries=self.num_quries, num_stages=self.num_stages, interpolate_scale_factor=config.interpolate_scale_factor, num_stacks=config.num_stacks, dim_head=64, heads=8, ff_mult=4,
            num_classes=self.num_classes, quantizer=self.quantizer, do_quantize=self.do_quantize, 
            finetune_codebook_only=config.finetune_codebook_only, upsample_mode=config.upsample_mode,
            use_self_attn = config.use_self_attn, semantic_label_dim=config.semantic_embed_dim, quant_use_seg=config.quant_use_seg
        )

    # def clone_vq_codebook(self, requires_grad):
    #     cloned_vqkd_embedding = copy_new_embedding(self.quantize.embedding_vqkd, requires_grad)
    #     cloned_vqgan_embedding = copy_new_embedding(self.quantize.embedding_vqgan, requires_grad)
    #     return (cloned_vqkd_embedding, cloned_vqgan_embedding)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def encode(self, image_features, do_quantize=False, mask_labels=None, class_labels=None, loss_type="dice_bce", semantic_labels=None, modality_label=None):
        """
        image_features: (b, h0, w0, d) - coarse feature map from ViT
        Returns:
            final_region_queries: (b, N, d)
            multi_scale_image_features: list[(b, h, w, d)] for each stage (including input)
        """
        #resize image_features from (b, h0*w0, d) to size (b, h0, w0, d)
        do_quantize = self.do_quantize and do_quantize
        if len(image_features.shape) == 2:
            image_features = image_features.unsqueeze(0)
        b, L, d = image_features.shape
        h0 = int(math.sqrt(L))
        # image_features = image_features.resize_(b, h0, h0, d)
        image_features = image_features.reshape(b, h0, h0, d).contiguous()
        outputs = self.region_perceiver(
            image_features, mask_labels=mask_labels, class_labels=class_labels, modality_label=modality_label, loss_type=loss_type, 
            do_quantize=do_quantize, semantic_labels=semantic_labels
        )
        # Unpack hierarchical outputs
        current_region_queries, multi_scale_image_features, seg_logits, class_logits, dice_loss, bce_loss, cls_loss, aux_outputs, \
        hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, quantization_losses, total_quantization_loss, \
        dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info, semantic_loss = outputs
        # current_region_queries, multi_scale_image_features, seg_logits, class_logits, dice_loss, bce_loss, cls_loss, aux_outputs = outputs
        quant_regions, emb_loss, info = None, None, None
        # hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, quantization_losses, total_quantization_loss = None, None, None, None, None, None
        # if do_quantize:
        #     quant_regions, emb_loss, info = self.quantize(current_region_queries)
        # perform modality-routed quantization here using current_region_queries if requested
        # if do_quantize and (self.quantizer is not None) and (current_region_queries is not None):
        #     # modality_label expected as list-like of length B or Tensor of indices
        #     q_modality_label = modality_label
        #     # If no explicit modality label provided, try to leave None (ModalitySpecificQuantizer will fallback)
        #     try:
        #         quant_regions, emb_loss, info = self.quantizer(current_region_queries, modality_label=q_modality_label, return_infos=True)
        #     except TypeError:
        #         # Some quantizers might not support return_infos kwarg
        #         quant_regions, emb_loss = self.quantizer(current_region_queries, modality_label=q_modality_label)
        #         info = None
        return current_region_queries, multi_scale_image_features, dice_loss, bce_loss, cls_loss, quant_regions, emb_loss, info, seg_logits, \
            class_logits, hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, quantization_losses, total_quantization_loss, \
            dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info, semantic_loss

    def decode_mask(self, quant, mask_labels=None):
        image_features = self.multi_scale_image_features[-1]
        mask_recon, seg_loss = self.region_perceiver.decode_mask(quant, image_features, mask_labels=mask_labels)

        # if self.teacher == 'siglip_384':
            # vqgan_recon = F.interpolate(vqgan_recon, size=(384, 384), mode='bicubic')

        dec = mask_recon
        return dec, seg_loss

    def forward(self, input, mask_labels=None, class_labels=None, do_quantize=False, loss_type="dice_bce", semantic_labels=None, llm_training=False, modality_label=None):
        # input: image
        # embed the image using medical clip model
        with torch.no_grad():
            # inputs = self.unimed_preprocess(input).to("cuda").unsqueeze(0)
            inputs = input.to("cuda")
            vision_output = self.image_encoder.forward_intermediates(inputs)
            # keep CLS token (index 0) for modality prediction, patch tokens at 1:
            # cls_token = vision_output['image_intermediates'][-1][:, 0, :]  # (B, D)
            # patch_features = vision_output['image_intermediates'][-1][:, 1:, :].squeeze(0)  # [576, D] for VIT-L
            last_feats = vision_output['image_intermediates'][-1]
            cls_token = last_feats[:, 0, :].detach().clone()          # (B, D)
            patch_features = last_feats[:, 1:, :].detach().clone()     # (B, L-1, D)

        # compute modality logits and optionally modality loss
        modality_logits = None
        modality_loss = None
        modality_pred_indices = None
        if getattr(self, 'modality_predictor', None) is not None:
            modality_logits = self.modality_predictor(cls_token.detach().to(self.modality_predictor.weight.device))
            modality_pred_indices = modality_logits.argmax(dim=-1).tolist()
            # print(modality_logits.size(), "logits size")
            # If no explicit modality_label provided, use predicted indices as routing keys
            if modality_label is None:
                modality_label = [str(i) for i in modality_pred_indices]
            else:
                # if ground-truth modality_label provided during training, compute cross-entropy loss
                if self.training:
                    # convert modality_label to tensor of ints
                    if not isinstance(modality_label, torch.Tensor):
                        try:
                            print(modality_label, "modality label")
                            labels_tensor = torch.tensor([int(x) for x in modality_label], dtype=torch.long, device=modality_logits.device)
                        except Exception:
                            labels_tensor = torch.tensor(modality_label, dtype=torch.long, device=modality_logits.device)
                    else:
                        labels_tensor = modality_label.to(modality_logits.device).long()
                    # print(modality_logits, labels_tensor, "logits and label")
                    modality_loss = self.modality_loss_fn(modality_logits, labels_tensor)
                    # print(modality_loss, "modality loss")

        # predict modality from cls token if not provided and predictor exists
        if modality_label is None and getattr(self, 'modality_predictor', None) is not None:
            with torch.no_grad():
                logits = self.modality_predictor(cls_token.to(self.modality_predictor.weight.device))
                indices = logits.argmax(dim=-1).tolist()
                # convert to string keys to match ModalitySpecificQuantizer keys
                modality_label = [str(i) for i in indices]

        current_region_queries, multi_scale_image_features, dice_loss, bce_loss, cls_loss, quant, diff, _, seg_logits, class_logits, \
        hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, quantization_losses, total_quantization_loss, \
        dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info, semantic_loss = self.encode(
            patch_features, do_quantize=do_quantize, mask_labels=mask_labels, class_labels=class_labels, loss_type=loss_type, semantic_labels=semantic_labels, modality_label=modality_label
        )
        dec = None
        self.multi_scale_image_features = multi_scale_image_features
        # attach modality prediction info to quantizer_info (if present) for logging
        # try:
        if quantizer_info is None:
            # print("quantizer_info is none")
            quantizer_info = {}
        quantizer_info['multi_scale_image_features'] = multi_scale_image_features
        if modality_logits is not None:
            # keep logits (tensor) and loss (tensor or None) in the info dict
            # quantizer_info = dict(quantizer_info) if not isinstance(quantizer_info, dict) else quantizer_info
            quantizer_info['modality_logits'] = modality_logits
            quantizer_info['modality_loss'] = modality_loss
            quantizer_info['modality_label_pred'] = modality_pred_indices
            quantizer_info['modality_label_gt'] = modality_label
                
        # except Exception:
        #     pass
        # incorporate modality classification loss into total_quantization_loss for unified reporting
        try:
            if modality_loss is not None:
                if total_quantization_loss is None:
                    total_quantization_loss = modality_loss
                else:
                    try:
                        total_quantization_loss = total_quantization_loss + modality_loss
                    except Exception:
                        print("add modality loss", modality_loss, total_quantization_loss)
                        total_quantization_loss = total_quantization_loss + modality_loss.to(total_quantization_loss.device)
        except Exception:
            pass
        # Optionally, return hierarchical losses/masks/codes for logging or analysis
        if not llm_training:
            return dec, diff, dice_loss, bce_loss, cls_loss, seg_logits, class_logits, hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, quantization_losses, total_quantization_loss, dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info, semantic_loss
        else:
            return patch_features, current_region_queries, quantizer_info
### shared quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio=0.0, l2_norm=True, show_usage=True, kmeans=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.kmeans_init = kmeans
        self.initted = False

        if self.kmeans_init:
            print("using kmeans init")
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim)
            self.embedding.weight.requires_grad = False
        else:
            print("no kmeans init")
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            if self.l2_norm:
                self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(131072)))

    def forward(self, z):
        # z: (b, n, d) or (b, d)
        orig_shape = z.shape
        if z.dim() == 2:
            z = z.unsqueeze(1)
        b, n, d = z.shape
        z_flattened = z.reshape(-1, d)
        if self.l2_norm:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        if self.kmeans_init and not self.initted:
            with torch.no_grad():
                z_flatteneds = [torch.zeros_like(z_flattened) for _ in range(torch.distributed.get_world_size())]
                dist.all_gather(z_flatteneds, z_flattened)
                combined_z_flatteneds = torch.cat(z_flatteneds, dim=0)
                print("combined_z_flatteneds.shape", combined_z_flatteneds.shape)
                self.embedding.init_embed_(combined_z_flatteneds)
                self.initted = True

        # Compute distances
        dists = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(embedding ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, embedding.t())
        )
        min_encoding_indices = torch.argmin(dists, dim=1)

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
        else:
            codebook_usage = 0

        encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        z_q = embedding[min_encoding_indices].view(b, n, d)

        # Losses
        vq_loss = torch.mean((z_q - z.detach()) ** 2)
        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
        entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-dists)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Reshape back to original shape if needed
        if orig_shape != z_q.shape:
            z_q = z_q.view(orig_shape)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None):
        embedding = self.embedding.weight
        if self.l2_norm:
            embedding = F.normalize(embedding, p=2, dim=-1)
        z_q = embedding[indices]  # (b*n, d) or (b, n, d)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q


def compute_entropy_loss(logits, dim=-1, eps=1e-12):
    # optional; keep your original if you already have one
    p = logits.log_softmax(dim=dim).exp()
    return -(p * (p.clamp_min(eps)).log()).sum(dim=dim).mean()



# --- helpers (tiny) ---

def _cosine_scores(z, emb):
    z = F.normalize(z, p=2, dim=-1)
    emb = F.normalize(emb, p=2, dim=-1)
    return z @ emb.t()  # (M,K), higher is better

def _topk_soft_assign(scores, k=2, tau=1.0):
    # scores: (M,K), higher is better
    k = min(k, scores.size(1))
    vals, idx = torch.topk(scores, k=k, dim=1)
    mask = torch.zeros_like(scores).scatter_(1, idx, 1.0)
    logits = torch.zeros_like(scores).scatter_(1, idx, vals / max(tau, 1e-6))
    prob = F.softmax(logits, dim=1)
    return prob  # (M,K)

class VectorQuantizerST(nn.Module):
    def __init__(
        self,
        n_e, e_dim, beta,
        entropy_loss_ratio=1.0,
        l2_norm=True,
        # NEW: distance options
        use_cosine=False, temperature=1.0,
        # NEW: safer kmeans init
        kmeans=False, kmeans_buffer_batches=8,
        # (unchanged interface)
        show_usage=True,
        route_grad_to: str = "both",
        # NEW: forbid EMA+grads mixing
        allow_codebook_grad: bool = False,
        # NEW: soft-assign warm-up
        use_soft_assign=True, soft_topk=2, soft_assign_steps=10000,
    ):
        super().__init__()
        assert route_grad_to in {"encoder", "embedding", "both", "none"}
        self.n_e, self.e_dim = n_e, e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        # ---- distance knobs ----
        self.use_cosine = use_cosine
        self.tau = temperature

        # ---- init knobs ----
        self.kmeans_init = kmeans
        self.kmeans_buffer_batches = kmeans_buffer_batches

        # ---- grad routing ----
        self.route_grad_to = route_grad_to
        self.allow_codebook_grad = allow_codebook_grad

        # ---- soft-assign warm-up ----
        self.use_soft_assign = use_soft_assign
        self.soft_topk = soft_topk
        self.soft_assign_steps = soft_assign_steps

        # ---- codebook ----
        if self.kmeans_init:
            # assume EMA codebook implementation
            self.embedding = EmbeddingEMA(self.n_e, self.e_dim)
            self.embedding.weight.requires_grad_(False)
            allow_codebook_grad = False
            self.allow_codebook_grad = False
            assert not allow_codebook_grad, "EMA codebook must not receive grads. Set allow_codebook_grad=False."
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            nn.init.uniform_(self.embedding.weight, -1.0 / self.n_e, 1.0 / self.n_e)
            print("Codebook initialized with uniform distribution, allow gradient:", allow_codebook_grad)
            if allow_codebook_grad:
                self.embedding.weight.requires_grad_(True)
            else:
                self.embedding.weight.requires_grad_(False)

        # ---- state / buffers ----
        self.register_buffer("initted", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        if show_usage:
            self.register_buffer("codebook_used", torch.empty(131072, dtype=torch.long).fill_(-1))
        self.register_buffer("ema_usage", torch.ones(n_e) / n_e)
        self.ema_usage_decay = 0.99   # decay for usage EMA 
        # kmeans sample buffer (python list, small)
        self._km_buf = []

    @torch.no_grad()
    def _maybe_kmeans_init(self, z_flat):
        if not self.training:
            return
        if not self.kmeans_init or bool(self.initted.item()):
            return
        # collect a few batches locally
        self._km_buf.append(z_flat.detach().cpu())
        if len(self._km_buf) < self.kmeans_buffer_batches:
            return
        samples = torch.cat(self._km_buf, dim=0)
        # (optional) could all_gather here if desired
        self.embedding.init_embed_(samples)
        self._km_buf.clear()
        self.initted.fill_(1)

    def forward(self, z):
        # z: (B,N,D) or (B,D)
        orig_shape = z.shape
        if z.dim() == 2:
            z = z.unsqueeze(1)
        B, N, D = z.shape
        M = B * N

        z_flat = z.view(M, D)
        if self.l2_norm:
            z_flat = F.normalize(z_flat, p=2, dim=-1)

        # k-means init from buffered batches
        self._maybe_kmeans_init(z_flat)

        # codebook
        emb = self.embedding.weight
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=-1)

        # ---- compute scores with temperature ----
        if self.use_cosine:
            # higher is better
            scores = _cosine_scores(z_flat, emb) / max(self.tau, 1e-6)  # (M,K)
            dists = -scores
        else:
            # Euclidean, lower is better; scale by tau
            z2 = (z_flat**2).sum(dim=1, keepdim=True)        # (M,1)
            e2 = (emb**2).sum(dim=1).unsqueeze(0)            # (1,K)
            dists = (z2 + e2 - 2.0 * (z_flat @ emb.t())) / max(self.tau, 1e-6)
            scores = -dists

        # ---- soft-assign warm-up (first K steps) ----
        use_soft_now = self.use_soft_assign and self.training and (self.step.item() < self.soft_assign_steps)
        if use_soft_now:
            # print("use soft!")
            # prob = _topk_soft_assign(scores, k=self.soft_topk, tau=self.tau)  # (M,K)
            prob = F.softmax(scores / self.tau, dim=1)
            hard_idx = prob.argmax(dim=1)                                     # (M,)
            z_q_soft = prob @ emb                                             # (M,D)
            z_q_hard = emb[hard_idx]
            # straight-through: fwd hard, bwd soft
            z_q_flat = z_q_hard.detach() + (z_q_soft - z_q_hard).detach().neg() + z_q_soft
            min_indices = hard_idx
        else:
            if self.use_cosine:
                min_indices = scores.argmax(dim=1)
            else:
                min_indices = dists.argmin(dim=1)
            z_q_flat = emb[min_indices]

        z_q = z_q_flat.view(B, N, D)

        # ---- losses ----
        vq_loss = torch.mean((z_q - z.detach())**2)
        commit_loss = self.beta * torch.mean((z_q.detach() - z)**2)

        
        # (optional) simple entropy proxy from scores
        # if self.entropy_loss_ratio > 0:
        #     p = F.softmax(scores, dim=1)
        #     entropy = -(p * (p.clamp_min(1e-8).log())).sum(dim=1).mean()
        #     entropy_loss = self.entropy_loss_ratio * entropy
        # else:
        #     entropy_loss = z_q.mean()*0

        # ---- EMA usage monitor + entropy loss ----
        if self.entropy_loss_ratio > 0 and self.training:
            with torch.no_grad():
                # histogram of current batch assignments
                batch_hist = torch.bincount(min_indices, minlength=self.n_e).float().to(z.device)
                batch_hist /= batch_hist.sum()

                # update global EMA usage
                self.ema_usage.mul_(self.ema_usage_decay).add_(
                    (1 - self.ema_usage_decay) * batch_hist
                )

            # normalized distribution
            usage_prob = self.ema_usage / self.ema_usage.sum()

            # entropy in [0,1]
            entropy = -(usage_prob * usage_prob.clamp_min(1e-8).log()).sum()
            entropy = entropy / math.log(self.n_e)

            entropy_loss = self.entropy_loss_ratio * entropy
        else:
            entropy_loss = torch.tensor(0.0, device=z.device)

        # ---- straight-through routing ----
        if self.route_grad_to == "encoder":
            z_q_st = z + (z_q - z).detach()
        elif self.route_grad_to == "embedding":
            z_q_st = z_q + (z - z.detach())
        elif self.route_grad_to == "both":
            z_q_st = z_q
        else:  # 'none'
            z_q_st = z.detach() + (z_q - z_q.detach())

        # ---- usage tracking (unchanged behavior) ----
        if self.show_usage and self.training:
            cur_len = min_indices.shape[0]
            # shift-left and append new indices
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_indices.detach()
            # (optional) you can compute perplexity externally if you like

        self.step += 1

        if orig_shape != z_q_st.shape:
            z_q_st = z_q_st.view(orig_shape)

        return (
            z_q_st,
            (vq_loss, commit_loss, entropy_loss, torch.tensor(0.0, device=z_q_st.device)),  # (keep tuple shape)
            # (None, None, min_indices.view(B, N))
            {
                "indices": min_indices.view(B, N),
                "codebook_used": self.codebook_used.clone() if self.show_usage else None,
                "assignment_probs": prob if use_soft_now else None  # if you want soft assignment info
            }
        )

    def get_codebook_entry(self, indices, shape=None):
        emb = self.embedding.weight
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=-1)
        z_q = emb[indices]
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        n_e, e_dim, beta,
        entropy_loss_ratio=0.0,
        l2_norm=True,
        # distance knobs
        use_cosine=False, temperature=1.0,
        # kmeans/EMA init
        kmeans=False, kmeans_buffer_batches=8,
        # usage monitor
        show_usage=True,
        # grad routing
        route_grad_to: str = "encoder",
        allow_codebook_grad: bool = False,
        # soft-assign warm-up
        use_soft_assign=True, soft_topk=2, soft_assign_steps=200,
        # EMA decay factors
        ema_decay=0.99, ema_eps=1e-5,
    ):
        super().__init__()
        assert route_grad_to in {"encoder", "embedding", "both", "none"}

        self.n_e, self.e_dim = n_e, e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        # distance knobs
        self.use_cosine = use_cosine
        self.tau = temperature

        # init knobs
        self.kmeans_init = kmeans
        self.kmeans_buffer_batches = kmeans_buffer_batches

        # grad routing
        self.route_grad_to = route_grad_to
        self.allow_codebook_grad = allow_codebook_grad

        # soft-assign warm-up
        self.use_soft_assign = use_soft_assign
        self.soft_topk = soft_topk
        self.soft_assign_steps = soft_assign_steps

        # EMA params
        self.decay = ema_decay
        self.eps = ema_eps

        # ---- EMA codebook ----
        embed = torch.randn(n_e, e_dim)
        if l2_norm:
            embed = F.normalize(embed, p=2, dim=-1)
        self.register_buffer("embed", embed)          # codebook vectors
        self.register_buffer("cluster_size", torch.zeros(n_e))
        self.register_buffer("embed_avg", embed.clone())

        # ---- state / buffers ----
        self.register_buffer("initted", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        if show_usage:
            self.register_buffer("codebook_used", torch.empty(131072, dtype=torch.long).fill_(-1))
            self.register_buffer("ema_usage", torch.ones(n_e) / n_e)

        # kmeans buffer
        self._km_buf = []

    @torch.no_grad()
    def _maybe_kmeans_init(self, z_flat):
        if not self.kmeans_init or bool(self.initted.item()):
            return
        self._km_buf.append(z_flat.detach().cpu())
        if len(self._km_buf) < self.kmeans_buffer_batches:
            return
        samples = torch.cat(self._km_buf, dim=0)
        # simple kmeans++ init could be plugged here
        idx = torch.randperm(samples.size(0))[: self.n_e]
        self.embed.copy_(samples[idx].to(self.embed.device))
        self.embed_avg.copy_(self.embed)
        self.cluster_size.zero_()
        self._km_buf.clear()
        self.initted.fill_(1)

    def forward(self, z):
        orig_shape = z.shape
        if z.dim() == 2:
            z = z.unsqueeze(1)
        B, N, D = z.shape
        M = B * N

        z_flat = z.view(M, D)
        if self.l2_norm:
            z_flat = F.normalize(z_flat, p=2, dim=-1)

        # maybe kmeans init
        self._maybe_kmeans_init(z_flat)

        # normalize codebook if needed
        emb = self.embed
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=-1)

        # compute distances/scores
        if self.use_cosine:
            scores = (z_flat @ emb.t()) / max(self.tau, 1e-6)
            dists = -scores
        else:
            z2 = (z_flat**2).sum(dim=1, keepdim=True)
            e2 = (emb**2).sum(dim=1).unsqueeze(0)
            dists = (z2 + e2 - 2.0 * (z_flat @ emb.t())) / max(self.tau, 1e-6)
            scores = -dists

        # assignment
        use_soft_now = self.use_soft_assign and self.training and (self.step.item() < self.soft_assign_steps)
        if use_soft_now:
            prob = F.softmax(scores / self.tau, dim=1)
            hard_idx = prob.argmax(dim=1)
            z_q_soft = prob @ emb
            z_q_hard = emb[hard_idx]
            z_q_flat = z_q_hard.detach() + (z_q_soft - z_q_hard).detach().neg() + z_q_soft
            min_indices = hard_idx
        else:
            min_indices = dists.argmin(dim=1)
            z_q_flat = emb[min_indices]
            prob = None

        z_q = z_q_flat.view(B, N, D)

        # losses
        vq_loss = torch.mean((z_q - z.detach()) ** 2)
        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        # ---- EMA codebook update ----
        with torch.no_grad():
            one_hot = F.one_hot(min_indices, self.n_e).type_as(z_flat)
            cluster_size = one_hot.sum(0)
            embed_sum = one_hot.t() @ z_flat

            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_e * self.eps)) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.copy_(embed_normalized)

        # ---- EMA usage monitor + entropy loss ----
        if self.entropy_loss_ratio > 0 and self.training:
            with torch.no_grad():
                batch_hist = torch.bincount(min_indices, minlength=self.n_e).float().to(z.device)
                batch_hist /= batch_hist.sum()
                self.ema_usage.mul_(self.decay).add_((1 - self.decay) * batch_hist)

            usage_prob = self.ema_usage / self.ema_usage.sum()
            entropy = -(usage_prob * usage_prob.clamp_min(1e-8).log()).sum()
            entropy = entropy / math.log(self.n_e)  # normalize to [0,1]
            entropy_loss = self.entropy_loss_ratio * entropy
        else:
            entropy_loss = torch.tensor(0.0, device=z.device)

        # straight-through routing
        if self.route_grad_to == "encoder":
            z_q_st = z + (z_q - z).detach()
        elif self.route_grad_to == "embedding":
            z_q_st = z_q + (z - z.detach())
        elif self.route_grad_to == "both":
            z_q_st = z_q
        else:
            z_q_st = z.detach() + (z_q - z_q.detach())

        # usage tracking
        if self.show_usage and self.training:
            cur_len = min_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_indices.detach()

        self.step += 1
        if orig_shape != z_q_st.shape:
            z_q_st = z_q_st.view(orig_shape)

        # perplexity monitor
        perplexity = (usage_prob.exp() if self.entropy_loss_ratio > 0 else None)

        return (
            z_q_st,
            (vq_loss, commit_loss, entropy_loss, torch.tensor(0.0, device=z_q_st.device)),
            {
                "indices": min_indices.view(B, N),
                "codebook_used": self.codebook_used.clone() if self.show_usage else None,
                "ema_usage": self.ema_usage.clone() if self.show_usage else None,
                "perplexity": perplexity,
                "assignment_probs": prob,
            }
        )

    def get_codebook_entry(self, indices, shape=None):
        emb = self.embed
        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=-1)
        z_q = emb[indices]
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q



#################################################################################
#                              VQ Model Configs                                 #
#################################################################################

def RegTokFunc(**kwargs):
    return RegTok(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))
    # return RegTok(ModelArgs(**kwargs))

VQ_models = {'RegTok': RegTokFunc}

class ModalitySpecificQuantizer(nn.Module):
    """Wrapper that maintains a codebook (quantizer) per modality.
    modalities: list of modality keys (strings) or integer count.
    Each modality maps to an instance of the existing VectorQuantizer/VectorQuantizerST.

    Forward accepts z: Tensor (B, ...), modality_label: list[str] or Tensor of modality indices/keys.
    Returns concatenated outputs and aggregated losses and infos.
    """
    def __init__(self, modality_keys, quantizer_factory, *qargs, **qkwargs):
        super().__init__()
        # modality_keys can be list of strings or int (num modalities)
        if isinstance(modality_keys, int):
            modality_keys = [str(i) for i in range(modality_keys)]
        self.modalities = list(modality_keys)
        self.codebooks = nn.ModuleDict()
        for m in self.modalities:
            # quantizer_factory is a callable returning a Module (e.g. VectorQuantizerST)
            self.codebooks[str(m)] = quantizer_factory(*qargs, **qkwargs)

    def forward(self, z, modality_label=None, return_infos=True, **kwargs):
        """
        z: Tensor of shape (B, C, ...) or (B, L, D) depending on quantizer API
        modality_label: list/iterable length B of modality keys or a tensor of indices
        return_infos: ignored (kept for compatibility) — always returns a triple like VectorQuantizerST
        """
        B = z.shape[0]
        if modality_label is None:
            # fallback: assign all to the first modality
            modality_label = [self.modalities[0]] * B
        # allow tensor of indices
        # print(modality_label, "modality label")
        if not isinstance(modality_label, (list, tuple)):
            try:
                modality_label = modality_label.tolist()
            except Exception:
                modality_label = [modality_label]
        outputs = []
        losses_vq = []
        infos = []
        # process sample-wise to route to modality-specific codebook
        for i in range(B):
            key = modality_label[i]
            key = str(key)
            if key not in self.codebooks:
                # fallback to first modality if unknown
                key = self.modalities[0]
            # assume per-sample z is z[i:i+1]
            quantizer = self.codebooks[key]
            out = quantizer(z[i:i+1], **kwargs)
            # expected out: (z_q, (vq_loss, commit_loss, entropy_loss, usage), info_dict)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                q_out, loss_tuple, info_dict = out
            elif isinstance(out, (tuple, list)) and len(out) == 2:
                q_out, loss_tuple = out
                info_dict = None
            else:
                q_out, loss_tuple, info_dict = out, None, None
            outputs.append(q_out)
            losses_vq.append(loss_tuple)
            infos.append(info_dict)
        # concatenate outputs along batch dim
        try:
            outputs = torch.cat(outputs, dim=0)
        except Exception:
            outputs = torch.stack(outputs, dim=0)
        device = outputs.device if isinstance(outputs, torch.Tensor) else torch.device('cpu')
        # aggregate losses (ignore None)
        vq_vals = [lv[0] for lv in losses_vq if (lv is not None and lv[0] is not None)]
        commit_vals = [lv[1] for lv in losses_vq if (lv is not None and lv[1] is not None)]
        entropy_vals = [lv[2] for lv in losses_vq if (lv is not None and lv[2] is not None)]
        usage_vals = [lv[3] for lv in losses_vq if (lv is not None and lv[3] is not None)]
        if len(vq_vals) > 0:
            vq_loss = torch.stack([v.to(device) for v in vq_vals], dim=0).mean()
        else:
            vq_loss = torch.tensor(0.0, device=device)
        if len(commit_vals) > 0:
            commit_loss = torch.stack([v.to(device) for v in commit_vals], dim=0).mean()
        else:
            commit_loss = torch.tensor(0.0, device=device)
        if len(entropy_vals) > 0:
            entropy_loss = torch.stack([v.to(device) for v in entropy_vals], dim=0).mean()
        else:
            entropy_loss = torch.tensor(0.0, device=device)
        if len(usage_vals) > 0:
            try:
                usage_mean = torch.stack([v.to(device) for v in usage_vals], dim=0).mean()
            except Exception:
                # some usage vals may be scalars
                usage_mean = torch.tensor(float(np.mean([float(v) for v in usage_vals])), device=device)
        else:
            usage_mean = torch.tensor(0.0, device=device)
        # build aggregated info: try to concatenate 'indices' if present
        indices_list = [info.get('indices') for info in infos if (info is not None and isinstance(info, dict) and 'indices' in info)]
        if len(indices_list) > 0:
            try:
                indices = torch.cat(indices_list, dim=0).to(device)
            except Exception:
                indices = torch.stack(indices_list, dim=0).to(device)
        else:
            indices = None
        # ensure modality_label is represented as list
        try:
            modalities_repr = list(modality_label)
        except Exception:
            modalities_repr = [modality_label]
        info_out = {
            'indices': indices,
            'per_sample_info': infos,
            'modality_labels': modalities_repr,
        }
        # print(info_out)
        # return in same shape as VectorQuantizerST
        return outputs, (vq_loss, commit_loss, entropy_loss, usage_mean), info_out


