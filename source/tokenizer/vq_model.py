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
    codebook_size: int = 128
    codebook_embed_dim: int = 16
    semantic_embed_dim: int = 768
    num_stages: int = 3
    num_stacks: int = 1
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 1.0
    entropy_loss_ratio: float = 0.0
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
            self.quantizer = VectorQuantizerST(n_e=config.codebook_size, e_dim=self.code_dim, beta=0.25, route_grad_to="embedding", allow_codebook_grad=allow_codebook_grad)

        ### load visual perceiver ###
        self.region_perceiver = RegionPerceiver(
            dim=1024, num_queries=self.num_quries, num_stages=self.num_stages, interpolate_scale_factor=config.interpolate_scale_factor, num_stacks=config.num_stacks, dim_head=64, heads=8, ff_mult=4,
            num_classes=self.num_classes, quantizer=self.quantizer, do_quantize=self.do_quantize, 
            finetune_codebook_only=config.finetune_codebook_only, upsample_mode=config.upsample_mode,
            use_self_attn = config.use_self_attn, semantic_label_dim=config.semantic_embed_dim
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

    def encode(self, image_features, do_quantize=False, mask_labels=None, class_labels=None, loss_type="dice_bce", semantic_labels=None):
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
        image_features = image_features.resize_(b, h0, h0, d)
        outputs = self.region_perceiver(
            image_features, mask_labels=mask_labels, class_labels=class_labels, loss_type=loss_type, 
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

    # def decode_code(self, code_b):
        # batch_size = code_b.size(0)
        # total_tokens = code_b.size(1)
        
        # current_total = 0
        # used_scales = []
        # for scale_size in self.scale_rq_layers:
        #     scale_tokens = scale_size ** 2
        #     if current_total + scale_tokens > total_tokens:
        #         break
        #     used_scales.append(scale_size)
        #     current_total += scale_tokens
        #     if current_total == total_tokens:
        #         break
        
        # if current_total != total_tokens:
        #     raise ValueError(
        #         f"Invalid code_b size {total_tokens}, "
        #         f"expected sum of scale tokens {current_total} "
        #         f"(scales: {used_scales})"
        #     )

        # quant_b = 0.0
        # current_pos = 0
        
        # for scale_size in used_scales:
        #     num_tokens = scale_size ** 2
        #     indices = code_b[:, current_pos:current_pos+num_tokens]
        #     current_pos += num_tokens            
        #     quant_this_scale = self.quantize.get_codebook_entry(indices)
        #     quant_this_scale = rearrange(quant_this_scale, 'b (h w) c -> b c h w', h=scale_size, w=scale_size)
        #     quant_this_scale = F.interpolate(
        #         quant_this_scale,
        #         size=(self.scale_rq_layers[-1], self.scale_rq_layers[-1]),
        #         mode='bicubic'
        #     )
        #     quant_b += quant_this_scale

        # dec = self.decode(quant_b)
        # return dec

    def forward(self, input, mask_labels=None, class_labels=None, do_quantize=False, loss_type="dice_bce", semantic_labels=None, llm_training=False):
        # input: image
        # embed the image using medical clip model
        with torch.no_grad():
            # inputs = self.unimed_preprocess(input).to("cuda").unsqueeze(0)
            inputs = input.to("cuda")
            vision_output = self.image_encoder.forward_intermediates(inputs)
            patch_features = vision_output['image_intermediates'][-1][:, 1:, :].squeeze(0)  # [576, D] for VIT-L

        current_region_queries, multi_scale_image_features, dice_loss, bce_loss, cls_loss, quant, diff, _, seg_logits, class_logits, \
        hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, quantization_losses, total_quantization_loss, \
        dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info, semantic_loss = self.encode(
            patch_features, do_quantize=do_quantize, mask_labels=mask_labels, class_labels=class_labels, loss_type=loss_type, semantic_labels=semantic_labels
        )
        dec = None
        self.multi_scale_image_features = multi_scale_image_features
        # if do_quantize:
        #     dec, seg_loss = self.decode_mask(quant, multi_scale_image_features[-1], mask_labels=mask_labels, return_loss=True)
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


# def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
#     flat_affinity = affinity.reshape(-1, affinity.shape[-1])
#     flat_affinity /= temperature
#     probs = F.softmax(flat_affinity, dim=-1)
#     log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
#     if loss_type == "softmax":
#         target_probs = probs
#     else:
#         raise ValueError("Entropy loss {} not supported".format(loss_type))
#     avg_probs = torch.mean(target_probs, dim=0)
#     avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
#     sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
#     loss = sample_entropy - avg_entropy
#     return loss


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
        entropy_loss_ratio=0.0,
        l2_norm=True,
        # NEW: distance options
        use_cosine=False, temperature=1.0,
        # NEW: safer kmeans init
        kmeans=False, kmeans_buffer_batches=8,
        # (unchanged interface)
        show_usage=True,
        route_grad_to: str = "encoder",
        # NEW: forbid EMA+grads mixing
        allow_codebook_grad: bool = False,
        # NEW: soft-assign warm-up
        use_soft_assign=False, soft_topk=2, soft_assign_steps=200,
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

        # kmeans sample buffer (python list, small)
        self._km_buf = []

    @torch.no_grad()
    def _maybe_kmeans_init(self, z_flat):
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
            prob = _topk_soft_assign(scores, k=self.soft_topk, tau=self.tau)  # (M,K)
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
        if self.entropy_loss_ratio > 0:
            p = F.softmax(scores, dim=1)
            entropy = -(p * (p.clamp_min(1e-8).log())).sum(dim=1).mean()
            entropy_loss = self.entropy_loss_ratio * entropy
        else:
            entropy_loss = z_q.mean()*0

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


# class VectorQuantizerST(nn.Module):
#     def __init__(self, n_e, e_dim, beta, entropy_loss_ratio=0.0,
#                  l2_norm=True, show_usage=True, kmeans=True,
#                  route_grad_to: str = "encoder",
#                  allow_codebook_grad: bool = False):  # <-- NEW
#         super().__init__()
#         assert route_grad_to in {"encoder", "embedding", "both", "none"}
#         self.allow_codebook_grad = allow_codebook_grad
#         self.n_e = n_e
#         self.e_dim = e_dim
#         self.beta = beta
#         self.entropy_loss_ratio = entropy_loss_ratio
#         self.l2_norm = l2_norm
#         self.show_usage = show_usage
#         self.kmeans_init = kmeans
#         self.initted = False
#         self.route_grad_to = route_grad_to  # <-- NEW

#         if self.kmeans_init:
#             print("using kmeans init")
#             self.embedding = EmbeddingEMA(self.n_e, self.e_dim)
#             self.embedding.weight.requires_grad = False
#         else:
#             print("no kmeans init")
#             self.embedding = nn.Embedding(self.n_e, self.e_dim)
#             self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
#             if self.l2_norm:
#                 self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        
        

#         if self.show_usage:
#             self.register_buffer("codebook_used", nn.Parameter(torch.zeros(131072)))

#     def forward(self, z):
#         # z: (b, n, d) or (b, d)
#         orig_shape = z.shape
#         if z.dim() == 2:
#             z = z.unsqueeze(1)
#         b, n, d = z.shape

#         z_flat = z.reshape(-1, d)
#         if self.l2_norm:
#             z_flat = F.normalize(z_flat, p=2, dim=-1)
#             embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
#         else:
#             embedding = self.embedding.weight

#         # (optional) kmeans init
#         if self.kmeans_init and not self.initted:
#             with torch.no_grad():
#                 world = dist.get_world_size() if dist.is_initialized() else 1
#                 if world > 1:
#                     z_all = [torch.zeros_like(z_flat) for _ in range(world)]
#                     dist.all_gather(z_all, z_flat)
#                     samples = torch.cat(z_all, dim=0)
#                 else:
#                     samples = z_flat
#                 print("kmeans init on", samples.shape)
#                 self.embedding.init_embed_(samples)
#                 self.initted = True
#                 # <-- enable grad on the codebook if we want seg/det to shape it
#                 if self.allow_codebook_grad:
#                     self.embedding.weight.requires_grad_(True)
#                     print(f"set gradient as true, {self.route_grad_to}")
#                 else:
#                     self.embedding.weight.requires_grad_(False)

#         # distances
#         dists = (
#             (z_flat ** 2).sum(dim=1, keepdim=True)
#             + (embedding ** 2).sum(dim=1)
#             - 2 * (z_flat @ embedding.t())
#         )
#         min_indices = torch.argmin(dists, dim=1)
#         # print("min_indices", min_indices.shape, min_indices.dtype, min_indices)
#         if self.show_usage and self.training:
#             cur_len = min_indices.shape[0]
#             self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
#             self.codebook_used[-cur_len:] = min_indices
#             codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
#         else:
#             codebook_usage = 0.0

#         z_q = embedding[min_indices].view(b, n, d)

#         # --- losses (same as your original) ---
#         # with torch.no_grad():
#         vq_loss = torch.mean((z_q - z.detach()) ** 2)
#         commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
#         entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-dists)

#         # --- switchable straight-through (ONLY CHANGE THAT MATTERS) ---
#         if self.route_grad_to == "encoder":
#             # original: grads go to encoder, not to embedding
#             z_q_st = z + (z_q - z).detach()
#         elif self.route_grad_to == "embedding":
#             # NEW: grads go to embedding, not to encoder (use when encoder is frozen)
#             z_q_st = z_q + (z - z.detach())
#         elif self.route_grad_to == "both":
#             # allow grads to flow into both (rarely needed, but handy)
#             z_q_st = z_q
#         else:  # 'none' – block both
#             z_q_st = z.detach() + (z_q - z_q.detach())

#         # reshape back if needed
#         if orig_shape != z_q_st.shape:
#             z_q_st = z_q_st.view(orig_shape)

#         return (
#             z_q_st,                                    # to your decoder / head
#             (vq_loss, commit_loss, entropy_loss, codebook_usage),
#             (None, None, min_indices.view(b, n))       # keep your original tuple shape
#         )

#     def get_codebook_entry(self, indices, shape=None):
#         embedding = self.embedding.weight
#         if self.l2_norm:
#             embedding = F.normalize(embedding, p=2, dim=-1)
#         z_q = embedding[indices]
#         if shape is not None:
#             z_q = z_q.view(shape)
#         return z_q


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################

def RegTokFunc(**kwargs):
    return RegTok(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))
    # return RegTok(ModelArgs(**kwargs))

VQ_models = {'RegTok': RegTokFunc}
