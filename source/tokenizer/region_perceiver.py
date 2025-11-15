# """
# Based on: https://github.com/lucidrains/flamingo-pytorch
# """

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn
# import scipy.optimize
import scipy
import math

import torch.nn.functional as F
from .criterion import hungarian_assign, compute_losses_with_indices
from .criterion import hungarian_assign_with_semantic


from transformers import AutoModelForCausalLM, AutoTokenizer

def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    ff = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    # Initialize linear layers
    for m in ff:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    return ff


def with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos

class PerceiverCrossAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # Initialization
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.constant_(self.norm_media.weight, 1.0)
        nn.init.constant_(self.norm_media.bias, 0.0)
        nn.init.constant_(self.norm_latents.weight, 1.0)
        nn.init.constant_(self.norm_latents.bias, 0.0)

    def forward(self, x, region_latents, pos=None, query_pos=None, attn_mask=None):
        """
        Args:
            x (torch.Tensor): image features, shape (b, T, n1, D)
            region_latents (torch.Tensor): latent features, shape (b, T, n2, D)
            pos: positional encoding for x, shape (b, T, n1, D)
            query_pos: positional encoding for region_latents, shape (b, T, n2, D)
            attn_mask (torch.Tensor or None): attention mask, shape (..., n2, n1)
        """
        if x.dtype != self.norm_media.weight.dtype:
            # print(x.dtype, self.norm_media.weight.dtype, "dnjak")
            x = x.to(self.norm_media.weight.dtype)
        x = self.norm_media(x)

        if region_latents.dtype != self.norm_latents.weight.dtype:
            # print(region_latents.dtype, self.norm_latents.weight.dtype, "dnjak")
            region_latents = region_latents.to(self.norm_latents.weight.dtype)
        region_latents = self.norm_latents(region_latents)

        h = self.heads

        # Add position embedding
        # print(region_latents.size(), query_pos.size(), "cross")
        # q = self.to_q(with_pos_embed(region_latents, query_pos))
        q = self.to_q(region_latents)
        # print(q.size(), "ehu")
        # kv_input = with_pos_embed(x, pos)
        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # print(sim.size(), "sim")
        # print(attn_mask.size() if attn_mask is not None else None, "attn mask")
        if attn_mask is not None:
            # Ensure attn_mask is broadcastable and does not mask all keys for any query
            mask_shape = attn_mask.shape
            sim_shape = sim.shape
            # Optionally, print or assert shapes for debugging
            assert all([m == s or m == 1 for m, s in zip(mask_shape[-2:], sim_shape[-2:])]), \
                f"attn_mask shape {mask_shape} not broadcastable to sim shape {sim_shape}"
            # Check for all-masked rows (all True for a query)
            all_masked = attn_mask.all(dim=-1)
            # if all_masked.any():
            #     print("[WARNING] Some queries have all keys masked in attention. This will cause NaNs.")
            sim = sim.masked_fill(attn_mask, float('-1e9'))  # Use -1e9 instead of -inf for numerical stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)



class PerceiverSelfAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # Initialization
        nn.init.xavier_uniform_(self.to_qkv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): latent features
                shape (b, t, n, D)
        """
        x = self.norm(x)
        h = self.heads
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class RegionDecoderDualAttention(nn.Module):
    def __init__(self, dim=512, depth=3, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # Attention 1: region tokens query image features
                        PerceiverCrossAttention(dim=dim, dim_head=dim_head, heads=heads),
                        nn.LayerNorm(dim),
                        # Attention 2: image features query region tokens (swap Q and K/V)
                        PerceiverCrossAttention(dim=dim, dim_head=dim_head, heads=heads),
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, mult=ff_mult),
                        nn.LayerNorm(dim),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)
        # LayerNorm initialization
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)
        for layer in self.layers:
            for m in layer:
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, region_latents, x, query_pos):
        """
        Args:
            region_latents: (b, n, D)
            query_pos: (b, n, D)
            x: list of image features [(b, v, D)] for each layer
        Returns:
            (b, n, D)
        """
        region_latents = region_latents.unsqueeze(1)
        img_feats = [xi.unsqueeze(1) for xi in x]
        for i, (attn1, norm1, attn2, norm2, ff, ff_norm) in enumerate(self.layers):
            # Attention 1: region tokens query image features
            if query_pos is not None:
                region_latents = region_latents + query_pos.unsqueeze(1)
            region_latents = attn1(img_feats[i], region_latents) + region_latents
            region_latents = norm1(region_latents)
            # Attention 2: image features query region tokens (swap Q and K/V)
            img_feats[i] = attn2(region_latents, img_feats[i]) + img_feats[i]
            img_feats[i] = norm2(img_feats[i])
            # Feedforward on region tokens
            region_latents = ff(region_latents) + region_latents
            region_latents = ff_norm(region_latents)
        return self.norm(region_latents[:, 0])

class SegmentationHead(nn.Module):
    """
    Segmentation and classification head: computes similarity between region queries and per-pixel features,
    and predicts class logits for each region query.
    """
    def __init__(self, dim, num_classes=None, num_heads=None):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.num_classes = num_classes
        self.num_heads = 0 if num_heads is None else num_heads
        if num_classes is not None:
            self.class_embed = nn.Linear(dim, num_classes + 1)
            nn.init.xavier_uniform_(self.class_embed.weight)
            nn.init.constant_(self.class_embed.bias, 0.0)
        else:
            self.class_embed = None
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, region_tokens, image_features, output_mask=True, use_interpolate=True):
        """
        region_tokens: (b, n, d)
        image_features: (b, h, w, d)
        Returns:
            segmentation_logits: (b, n, h, w)
            class_logits: (b, n, num_classes) if num_classes is not None else None
            attn_mask: (b, num_heads, 1, n, h*w) for masked attention
            interp_attn_mask: (b, num_heads, 1, n, 4*h*w) for next layer
        """
        b, n, d = region_tokens.shape
        _, h, w, _ = image_features.shape
        region_tokens_proj = self.proj(region_tokens)  # (b, n, d)
        image_features_flat = image_features.view(b, h * w, d)  # (b, hw, d)

        # region_tokens_proj = F.normalize(region_tokens_proj, p=2, dim=-1)
        # image_features_flat = F.normalize(image_features_flat, p=2, dim=-1)

        # Clamp extreme values to avoid NaNs
        # region_tokens_proj = torch.clamp(region_tokens_proj, -50, 50)
        # image_features_flat = torch.clamp(image_features_flat, -50, 50)
        # Compute similarity
        if image_features_flat.dtype != region_tokens_proj.dtype:
            image_features_flat = image_features_flat.to(region_tokens_proj.dtype)
        sim = torch.einsum('bnd,bmd->bnm', region_tokens_proj, image_features_flat)  # (b, n, hw)
        outputs_mask = sim.view(b, n, h, w)

        # Debug: print mask logits statistics
        # print("[SegmentationHead] outputs_mask stats:",
        #       "min:", outputs_mask.min().item(),
        #       "max:", outputs_mask.max().item(),
        #       "mean:", outputs_mask.mean().item(),
        #       "has_nan:", torch.isnan(outputs_mask).any().item())

        # Attention mask for current layer: (b, num_heads, 1, n, h*w)
        mask_logits = outputs_mask  # (b, n, h, w)
        mask_logits_flat = mask_logits.view(b, n, h * w)  # (b, n, hw)
        attn_mask = mask_logits_flat.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, n, hw)
        attn_mask = attn_mask.repeat(1, self.num_heads, 1, 1, 1)  # (b, num_heads, 1, n, hw)
        attn_mask = (attn_mask.sigmoid() < 0.5).bool()
        attn_mask = attn_mask.detach()
        interp_attn_mask = None
        if output_mask and use_interpolate:
            # Interpolated mask for next layer: (b, n, 2*h, 2*w) -> (b, n, 4*h*w)
            interp_mask_logits = F.interpolate(mask_logits, size=(2*h, 2*w), mode="bilinear", align_corners=False)
            interp_mask_flat = interp_mask_logits.view(b, n, 4*h*w)  # (b, n, 4hw)
            interp_attn_mask = interp_mask_flat.unsqueeze(1).unsqueeze(2)  # (b, 1, 1, n, 4hw)
            interp_attn_mask = interp_attn_mask.repeat(1, self.num_heads, 1, 1, 1)  # (b, num_heads, 1, n, 4hw)
            interp_attn_mask = (interp_attn_mask.sigmoid() < 0.5).bool()
            interp_attn_mask = interp_attn_mask.detach()
        else:
            interp_attn_mask = attn_mask

        class_logits = self.class_embed(region_tokens) if self.class_embed is not None else None
        return outputs_mask, class_logits, attn_mask, interp_attn_mask


class ConvAdapter(nn.Module):
    """
    Lightweight upsampling adapter: 2x bilinear upsample + 3x3 conv.
    """
    def __init__(self, dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        # Initialization
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        # x: (b, h, w, d)
        x = x.permute(0, 3, 1, 2)  # (b, d, h, w)
        x = self.upsample(x)
        if x.dtype != self.conv.weight.dtype:
            x = x.to(self.conv.weight.dtype)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (b, h, w, d)
        return x



class QueryAwareAdapter(nn.Module):
    """
    Query-aware upsampling adapter.
    - 2x bilinear upsample
    - Per-pixel FiLM from region queries, routed by mask logits or cosine similarity.
    - Local refinement with depthwise+pointwise conv and LayerNorm.
    """
    def __init__(self, dim, use_masks=True, tau: float = 1.0, detach_masks: bool = False):
        super().__init__()
        self.use_masks = use_masks
        self.tau = tau
        self.detach_masks = detach_masks
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.to_gamma = nn.Linear(dim, dim, bias=False)
        self.to_beta = nn.Linear(dim, dim, bias=False)
        self.dw_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_ln = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.to_gamma.weight)
        nn.init.xavier_uniform_(self.to_beta.weight)

    def _pixel_query_weights(self, x_up, queries, prev_mask_logits=None):
        """
        x_up: (B, H2, W2, D)
        queries: (B, N, D)
        prev_mask_logits: (B, N, H?) -> will be resized to (B, N, H2, W2)
        returns alpha: (B, H2, W2, N)  [softmax over N at each pixel]
        """
        B, H2, W2, D = x_up.shape

        if self.use_masks and (prev_mask_logits is not None):
            masks = prev_mask_logits
            # optional: stop grad to stabilize early training
            if self.detach_masks:
                masks = masks.detach()
            # upsample masks to match x_up spatial size if needed
            if masks.shape[-2:] != (H2, W2):
                masks = F.interpolate(
                    masks, size=(H2, W2), mode='bilinear', align_corners=False
                )
            # convert logits -> per-pixel query weights
            alpha = (masks / self.tau).permute(0, 2, 3, 1).softmax(dim=-1)  # (B,H2,W2,N)
            return alpha

        # fallback: cosine similarity between queries and *upsampled* pixels
        qn = F.normalize(queries, dim=-1)                    # (B,N,D)
        xn = F.normalize(x_up.view(B, H2 * W2, D), dim=-1)   # (B,H2W2,D)
        sim = torch.einsum('bnd,bmd->bnm', qn, xn)           # (B,N,H2W2)
        alpha = sim.softmax(dim=1).transpose(1, 2).view(B, H2, W2, -1)
        return alpha  # (B,H2,W2,N)

    def forward(self, x, queries, prev_mask_logits=None):
        # x: (B,H,W,D), queries: (B,N,D)
        B, H, W, D = x.shape

        # 1) upsample features
        x_up = self.up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # (B,2H,2W,D)

        # 2) per-pixel query weights
        alpha = self._pixel_query_weights(x_up, queries, prev_mask_logits)  # (B,2H,2W,N)

        # 3) FiLM params from queries, aggregated by alpha
        gamma_q = self.to_gamma(queries)  # (B,N,D)
        beta_q  = self.to_beta(queries)   # (B,N,D)
        gamma = torch.einsum('bhwn,bnd->bhwd', alpha, gamma_q)  # (B,2H,2W,D)
        beta  = torch.einsum('bhwn,bnd->bhwd', alpha, beta_q)   # (B,2H,2W,D)

        # 4) apply FiLM
        y = (1.0 + gamma) * x_up + beta                           # (B,2H,2W,D)

        # 5) local conv refinement (DW+PW in NCHW), then LN in NHWC
        y_nchw = y.permute(0, 3, 1, 2).contiguous()               # (B,D,2H,2W)
        y_nchw = self.pw_conv(F.gelu(self.dw_conv(y_nchw)))                 # (B,D,2H,2W)
        y = y_nchw.permute(0, 2, 3, 1).contiguous()               # (B,2H,2W,D)
        y = self.out_ln(y)                                        # LN over channel D (NHWC)

        return y

class QueryAwareConvAdapter(nn.Module):
    """
    Drop-in: 2x bilinear upsample + (optional) query-aware FiLM + 3x3 conv.
    Usage:
      y = adapter(x)                                 # plain (your original)
      y = adapter(x, queries=Q, prev_mask_logits=M)  # query-aware (preferred)
    Shapes:
      x:  (B,H,W,D)
      Q:  (B,N,D)
      M:  (B,N,H,W)  -- optional, logits at current stage
      out:(B,2H,2W,D)
    """
    def __init__(self, dim, tau: float = 1.0, use_masks: bool = True, detach_masks: bool = False):
        super().__init__()
        self.tau = tau
        self.use_masks = use_masks
        self.detach_masks = detach_masks

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.to_gamma = nn.Linear(dim, dim, bias=False)
        self.to_beta  = nn.Linear(dim, dim, bias=False)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

        nn.init.xavier_uniform_(self.to_gamma.weight)
        nn.init.xavier_uniform_(self.to_beta.weight)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    @torch.no_grad()
    def _alpha_from_masks(self, masks, H2, W2):
        # masks: (B,N,H,W) -> (B,N,2H,2W) -> (B,2H,2W,N), softmax over N
        if masks.shape[-2:] != (H2, W2):
            masks = F.interpolate(masks, size=(H2, W2), mode='bilinear', align_corners=False)
        alpha = (masks / self.tau).permute(0, 2, 3, 1).softmax(dim=-1)
        return alpha

    def _alpha_from_cosine(self, x_up, queries):
        # x_up: (B,H2,W2,D), queries: (B,N,D) -> (B,2H,2W,N)
        B, H2, W2, D = x_up.shape
        xn = F.normalize(x_up.view(B, H2 * W2, D), dim=-1)      # (B,HW,D)
        qn = F.normalize(queries, dim=-1)                       # (B,N,D)
        sim = torch.einsum('bmd,bnd->bmn', xn, qn)              # (B,HW,N)
        alpha = (sim / self.tau).softmax(dim=-1).view(B, H2, W2, -1)
        return alpha

    def forward(self, x, queries: torch.Tensor = None, prev_mask_logits: torch.Tensor = None):
        # x: (B,H,W,D)
        B, H, W, D = x.shape

        # 1) upsample features
        x_up = self.upsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # (B,2H,2W,D)

        # 2) optionally compute query-aware FiLM
        if (queries is None) or (queries.numel() == 0):
            y = x_up  # plain path
        else:
            H2, W2 = 2 * H, 2 * W
            if self.use_masks and (prev_mask_logits is not None):
                masks = prev_mask_logits.detach() if self.detach_masks else prev_mask_logits
                alpha = self._alpha_from_masks(masks, H2, W2)           # (B,2H,2W,N)
            else:
                alpha = self._alpha_from_cosine(x_up, queries)          # (B,2H,2W,N)

            gamma_q = self.to_gamma(queries)  # (B,N,D)
            beta_q  = self.to_beta(queries)   # (B,N,D)
            # per-pixel FiLM params
            gamma = torch.einsum('bhwn,bnd->bhwd', alpha, gamma_q)  # (B,2H,2W,D)
            beta  = torch.einsum('bhwn,bnd->bhwd',  alpha, beta_q)  # (B,2H,2W,D)
            y = (1.0 + gamma) * x_up + beta                         # (B,2H,2W,D)

        # 3) light smoothing (same as your original conv)
        y = self.conv(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return y

class BiDirectionalBlock(nn.Module):
    """
    Bi-directional transformer block for region-to-image and image-to-region attention.
    """
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4, use_self_attn=False):
        super().__init__()
        self.use_self_attn = use_self_attn
        # Region-to-image: region queries attend to image features
        self.region_to_image = PerceiverCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.region_to_image_norm = nn.LayerNorm(dim)
        # Self-attention for region queries (added)
        if self.use_self_attn:
            self.region_self_attn = PerceiverSelfAttention(dim=dim, dim_head=dim_head, heads=heads)
            self.region_self_attn_norm = nn.LayerNorm(dim)
        # Image-to-region: each pixel attends to region queries (not simple cross-attn)
        
        self.image_to_region = PerceiverCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.image_to_region_norm = nn.LayerNorm(dim)
        # Feedforward for region queries
        self.ff_region = FeedForward(dim, mult=ff_mult)
        self.ff_region_norm = nn.LayerNorm(dim)
        # Feedforward for image features
        self.ff_image = FeedForward(dim, mult=ff_mult)
        self.ff_image_norm = nn.LayerNorm(dim)
        # LayerNorm initialization
        if self.use_self_attn:
            for norm in [self.region_to_image_norm, self.region_self_attn_norm, self.image_to_region_norm, self.ff_region_norm, self.ff_image_norm]:
                nn.init.constant_(norm.weight, 1.0)
                nn.init.constant_(norm.bias, 0.0)
        else:
            for norm in [self.region_to_image_norm, self.image_to_region_norm, self.ff_region_norm, self.ff_image_norm]:
                nn.init.constant_(norm.weight, 1.0)
                nn.init.constant_(norm.bias, 0.0)

    def forward(self, region_queries, image_features, attn_mask=None, query_pos=None):
        """
        region_queries: (b, n, d)
        image_features: (b, h, w, d)
        attn_mask: optional attention mask for masked attention
        Returns:
            updated_region_queries: (b, n, d)
            updated_image_features: (b, h, w, d)
        """
        b, h, w, d = image_features.shape
        if query_pos is not None:
            region_queries = region_queries + query_pos
        # Region-to-image: queries attend to image features
        region_queries_ = self.region_to_image(
            image_features.view(b, 1, h * w, d), region_queries.unsqueeze(1), attn_mask=attn_mask
        ) + region_queries.unsqueeze(1)
        region_queries_ = self.region_to_image_norm(region_queries_)
        region_queries_ = region_queries_.squeeze(1)
        # Self-attention for region queries
        if self.use_self_attn:
            region_queries_ = self.region_self_attn(region_queries_.unsqueeze(1)).squeeze(1) + region_queries_
            region_queries_ = self.region_self_attn_norm(region_queries_)
        # Feedforward for region queries
        region_queries_ = self.ff_region(region_queries_) + region_queries_
        region_queries_ = self.ff_region_norm(region_queries_)
        # Image-to-region: each pixel attends to region queries
        image_features_flat = image_features.view(b, h * w, d)
        image_features_ = self.image_to_region(
            region_queries_.unsqueeze(1), image_features_flat.unsqueeze(1), attn_mask=None
        ) + image_features_flat.unsqueeze(1)
        image_features_ = self.image_to_region_norm(image_features_)
        image_features_ = image_features_.squeeze(1)
        image_features_ = self.ff_image(image_features_) + image_features_
        image_features_ = self.ff_image_norm(image_features_)
        image_features_ = image_features_.view(b, h, w, d)
        return region_queries_, image_features_

class StackedBiDirectionalBlock(nn.Module):
    def __init__(self, num_stacks, dim, dim_head=64, heads=8, ff_mult=4, use_self_attn=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiDirectionalBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, use_self_attn=use_self_attn)
            for _ in range(num_stacks)
        ])

    def forward(self, region_queries, image_features, attn_mask=None, query_pos=None):
        for block in self.blocks:
            region_queries, image_features = block(
                region_queries, image_features, attn_mask=attn_mask, query_pos=query_pos
            )
        return region_queries, image_features

class ConvFFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hid = dim * mult
        self.fc1 = nn.Linear(dim, hid, bias=False)
        self.act = nn.GELU()
        self.dw  = nn.Conv2d(hid, hid, 3, padding=1, groups=hid, bias=False)
        self.fc2 = nn.Linear(hid, dim, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, H, W):  # x: (B, HW, D)
        B, HW, _ = x.shape
        h = self.act(self.fc1(x))
        h2 = h.view(B, H, W, -1).permute(0,3,1,2)
        h2 = self.dw(h2).permute(0,2,3,1).contiguous().view(B, HW, -1)
        return self.fc2(h + h2)

class LightweightRegionDecoder(nn.Module):
    """
    Lightweight decoder for reconstructing queries and masks after quantization.
    Fuses integrated_codes with image features (optionally multi-scale).
    Uses positional encoding for image features.
    Uses previous layer's attention mask prediction for masked attention.
    """
    def __init__(self, dim, code_dim, use_seg=False, num_layers=3, use_multiscale=False, num_classes=None, num_heads=8, max_queries=512):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_seg = use_seg
        # Use a 2-layer MLP with GELU and LayerNorm for from_code
        if code_dim != dim:
            self.from_code = nn.Sequential(
                nn.Linear(code_dim, dim, bias=True),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim, bias=True),
                nn.GELU(),
                nn.LayerNorm(dim)
            )
        else:
            self.from_code = nn.Identity()
        if self.use_seg:
            self.segmentation_cls_head_quant = SegmentationHead(dim=dim, num_classes=num_classes)

    def forward(self, codes, image_features=None, multi_scale_image_features=None, interp_attn_mask=None):
        # codes: (b, n, d)
        # image_features: (b, h, w, d) -- final image features from RegionPerceiver
        region_latents = self.from_code(codes)
        image_feats_last_layer = None
        if self.use_seg:
            if multi_scale_image_features is not None:
                image_feats_last_layer = multi_scale_image_features[-1]
            else:
                image_feats_last_layer = image_features
        
        return region_latents, image_feats_last_layer

class PositionEmbeddingSine(nn.Module):
    """
    Sine-cosine positional encoding for image features of shape (b, h, w, d).
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        # x: (b, h, w, d)
        b, h, w, _ = x.shape
        device = x.device
        # mask: (b, h, w) or None
        if mask is None:
            mask = torch.zeros((b, h, w), device=device, dtype=torch.bool)
        not_mask = ~mask  # (b, h, w)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (b, h, w, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t  # (b, h, w, num_pos_feats)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # (b, h, w, 2*num_pos_feats)
        # If d > 2*num_pos_feats, pad zeros; if d < 2*num_pos_feats, truncate
        d = x.shape[-1]
        if pos.shape[-1] < d:
            pad = torch.zeros((b, h, w, d - pos.shape[-1]), device=device, dtype=pos.dtype)
            pos = torch.cat([pos, pad], dim=-1)
        elif pos.shape[-1] > d:
            pos = pos[..., :d]
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class RegionPerceiver(nn.Module):
    """
    Region Perceiver with Multi-scale Understanding and Iterative Upsampling.
    Outputs intermediate masks for masked attention and deep supervision.
    Supports hierarchical quantization and coarse-to-fine supervision.
    """
    def __init__(self, dim=512, num_queries=16, num_stacks=1, num_stages=2, dim_head=64, heads=8, ff_mult=4, num_classes=None, do_quantize=False, 
                 quantizer=None, quantize_intermediate=False, finetune_codebook_only=False, upsample_mode: str = "conv",
                 use_self_attn = False, quant_use_seg=False, *args, **kwargs):
        super().__init__()
        self.num_queries = num_queries
        self.num_stacks = num_stacks
        print("Number of stacks:", num_stacks)
        self.dim = dim
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.quantizer = quantizer  # <-- add quantizer for hierarchical codes
        if quantizer is not None:
            if hasattr(quantizer, "e_dim"):
                self.code_dim = quantizer.e_dim if quantizer is not None else -1
            else:
                self.code_dim = quantizer.codebooks["0"].e_dim
        else:
            self.code_dim = -1
        self.do_quantize = do_quantize
        self.quantize_intermediate = quantize_intermediate  # <-- control intermediate quantization
        self.finetune_codebook_only = finetune_codebook_only
        self.upsample_mode = upsample_mode
        print("Upsample mode:", upsample_mode)

        # Learnable region queries
        self.region_queries = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.xavier_uniform_(self.region_queries)
        self.region_position_embed = nn.Embedding(num_queries, dim)
        nn.init.xavier_uniform_(self.region_position_embed.weight)

        # Initial cross-attention for coarse features
        self.init_cross_attn = PerceiverCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.init_cross_norm = nn.LayerNorm(dim)
        self.init_ff = FeedForward(dim=dim, mult=ff_mult)
        self.init_ff_norm = nn.LayerNorm(dim)

        # Iterative upsampling adapters and bi-directional blocks
        if self.upsample_mode == "query":
            self.query_adapters = nn.ModuleList([QueryAwareConvAdapter(dim, use_masks=True) for _ in range(num_stages)])
        else:
            self.conv_adapters = nn.ModuleList([ConvAdapter(dim) for _ in range(num_stages)])
        # self.conv_adapters = nn.ModuleList([ConvAdapter(dim) for _ in range(num_stages)])
        # if self.num_stacks == 1:
        self.bi_blocks = nn.ModuleList([
            BiDirectionalBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, use_self_attn=use_self_attn)
            for _ in range(num_stages * self.num_stacks)
        ])
        # else:
        #     self.bi_blocks = nn.ModuleList([
        #         StackedBiDirectionalBlock(
        #             num_stacks=num_stacks, dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, use_self_attn=use_self_attn
        #         )
        #         for _ in range(num_stages)
        #     ])

        # Positional encoding for image features (like Mask2Former)
        N_steps = dim // 2
        self.position_encoding = PositionEmbeddingSine(N_steps, normalize=True)
        self.segmentation_cls_head = SegmentationHead(dim=dim, num_classes=num_classes, num_heads=heads)

        # A lightweight causal attention layer for modeling the codes from coarse to fine
        self.code_integration = None
        self.code_decoder, self.to_code = None, None
        if self.do_quantize:
            # Use a 2-layer MLP with GELU and LayerNorm for to_code
            if self.dim != self.code_dim:
                self.to_code = nn.Sequential(
                    nn.Linear(self.dim, self.dim, bias=True),
                    nn.GELU(),
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.code_dim, bias=True),
                    nn.GELU(),
                    nn.LayerNorm(self.code_dim)
                )
            else:
                self.to_code = nn.Identity()
            self.code_decoder = LightweightRegionDecoder(
                dim=dim, code_dim=self.code_dim, use_seg=quant_use_seg, num_classes=num_classes
            )
            print(num_classes, "num_classes")
        self.semantic_proj = None

        self.image_features_temp = None

    def decode_mask(self, region_codes):

        outputs = self.forward(image_features=self.image_features_temp, region_queries=region_codes)

        (
            current_region_queries, multi_scale_image_features, seg_logits_normal, class_logits_normal,
            dice_loss, bce_loss, cls_loss,
            aux_outputs,
            hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses,
            quantization_losses, total_quantization_loss,
            dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant,
            total_recon_loss, quantizer_info,
            semantic_loss
        ) = outputs

        return seg_logits_normal, class_logits_normal

    def forward(self, image_features, region_queries=None, mask_labels=None, class_labels=None, modality_label=None, loss_type="dice_bce", mask=None, do_quantize=False, deep_supervision=False, eos_coef=0.1,
                vq_coef=1.0, commit_coef=0.25, recon_coef=1.0, entropy_coef=0.01, semantic_labels=None, semantic_temperature=1.0):
        self.do_quantize = do_quantize and self.do_quantize
        """
        image_features: (b, h0, w0, d) - coarse feature map from ViT
        mask_labels: list of [ (num_gt, h, w) ] ground-truth masks for each image in batch (optional)
        mask: (b, h, w) or None, optional mask for masked attention
        deep_supervision: if True, output intermediate masks for each stage
        vq_coef, commit_coef, entropy_coef: coefficients for quantization losses
        Returns:
            final_region_queries: (b, N, d)
            multi_scale_image_features: list[(b, h, w, d)] for each stage (including input)
            seg_logits: (b, n, h, w) - segmentation logits at finest scale
            class_logits: (b, n, num_classes) if num_classes is not None else None
            seg_loss: segmentation loss (if mask_labels provided)
            aux_outputs: list of (seg_logits, class_logits) for deep supervision
            quantization_losses: dict of quantization losses (if do_quantize)
        """
        b, h0, w0, d = image_features.shape
        self.image_features_temp = image_features # save for decoding mask

        self.dtype = self.init_cross_norm.weight.dtype
        # print(image_features.dtype, self.region_queries.dtype, self.dtype, "bchw")
        if region_queries is None:
            region_queries = self.region_queries
            region_queries = region_queries.expand(b, -1, -1) # (b, N, d)
        if image_features.dtype != self.dtype:
            image_features = image_features.to(self.dtype)
            region_queries = region_queries.to(self.dtype)
        # print(region_queries.size(), "region_queries size")
        # Add positional encoding to image features (like Mask2Former)
        pos_enc = self.position_encoding(image_features)  # (b, h0, w0, d)
        image_features = image_features + pos_enc
        query_pos = self.region_position_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        region_queries_ = self.init_cross_attn(
            image_features.view(b, 1, h0 * w0, d), region_queries.unsqueeze(1), pos=None, query_pos=query_pos
        ) + region_queries.unsqueeze(1)

        region_queries_ = self.init_cross_norm(region_queries_)
        region_queries_ = region_queries_.squeeze(1)
        region_queries_ = self.init_ff(region_queries_) + region_queries_
        region_queries_ = self.init_ff_norm(region_queries_)

        multi_scale_image_features = [image_features]
        current_image_features = image_features
        current_region_queries = region_queries_

        seg_logits, class_logits, attn_mask, interp_attn_mask = self.segmentation_cls_head(current_region_queries, current_image_features)


        aux_outputs = []
        hierarchical_losses = []
        hierarchical_codes = []
        hierarchical_masks = []
        hierarchical_gt_masks = []
        quantization_losses = {
            "vq_loss": [],
            "commit_loss": [],
            "entropy_loss": [],
            "codebook_usage": [],
            "recon_loss": []
        }
        stage_seg_logits = None
        for stage in range(self.num_stages):
            # Upsample image features
            if self.upsample_mode == "query":
                # Use current stage's masks as FiLM routing (preferred; better SNR than cosine)
                prev_mask_logits = seg_logits if stage_seg_logits is None else stage_seg_logits  # (B, N, H, W) at current resolution
                upsampled_image_features = self.query_adapters[stage](
                    current_image_features,          # (B, H, W, D)
                    current_region_queries,          # (B, N, D)
                    prev_mask_logits=prev_mask_logits
                )
            else:
                upsampled_image_features = self.conv_adapters[stage](current_image_features)
            # upsampled_image_features = self.conv_adapters[stage](current_image_features)
            pos_enc = self.position_encoding(upsampled_image_features)
            upsampled_image_features = upsampled_image_features + pos_enc

            input_region_queries = current_region_queries
            input_image_features = upsampled_image_features

            for stack in range(self.num_stacks):
                block_idx = stage * self.num_stacks + stack
                # print(interp_attn_mask.size(), stage, stack, "interp mask")
                if input_region_queries.dtype != self.dtype:
                    input_region_queries = input_region_queries.to(self.dtype)
                if input_image_features.dtype != self.dtype:
                    input_image_features = input_image_features.to(self.dtype)
                input_region_queries, input_image_features = self.bi_blocks[block_idx](
                    input_region_queries, input_image_features, attn_mask=interp_attn_mask, query_pos=None
                )
                use_interpolate = True if stack == self.num_stacks-1 else False
                seg_logits, class_logits, attn_mask, interp_attn_mask = self.segmentation_cls_head(input_region_queries, input_image_features, use_interpolate=use_interpolate)
            current_region_queries = input_region_queries
            current_image_features = input_image_features
            multi_scale_image_features.append(current_image_features)
            aux_outputs.append((seg_logits, class_logits))
            

        # --- Integrate hierarchical codes with causal attention ---
        integrated_codes, quantizer_info = None, None
        if self.do_quantize and self.quantizer is not None:
            region_queries_for_quant = self.to_code(current_region_queries)
            try:
                integrated_codes, (vq_loss, commit_loss, entropy_loss, codebook_usage), quantizer_info = self.quantizer(region_queries_for_quant, modality_label=modality_label, return_infos=True)
                # print(quantizer_info)
            except Exception as e:
                integrated_codes, (vq_loss, commit_loss, entropy_loss, codebook_usage), quantizer_info = self.quantizer(region_queries_for_quant)

            quantization_losses["vq_loss"].append(vq_loss)
            quantization_losses["commit_loss"].append(commit_loss)
            quantization_losses["entropy_loss"].append(entropy_loss)
            quantization_losses["codebook_usage"].append(codebook_usage)
        else:
            integrated_codes = current_region_queries

        seg_logits_normal, class_logits_normal, _, _ = self.segmentation_cls_head(current_region_queries, current_image_features, output_mask=False)
        seg_logits_quant, class_logits_quant = None, None
        decoded_latents = integrated_codes  # <-- initialize as codes
        if self.do_quantize and self.quantizer is not None:
            # print("use code decoder for reconstruction")
            decoded_latents, decoded_image_features = self.code_decoder(integrated_codes, current_image_features, multi_scale_image_features)
            if decoded_image_features is not None:
                seg_logits_quant, class_logits_quant, _, _ = self.code_decoder.segmentation_cls_head_quant(decoded_latents, decoded_image_features, output_mask=False)
            # seg_logits_quant, class_logits_quant, _, _ = self.segmentation_cls_head(decoded_latents, decoded_image_features, output_mask=False)
        
        # --- Distillation / Reconstruction Loss ---
        # Compute reconstruction loss between decoded latents and original region queries (L2)
        recon_loss = None
        if decoded_latents is not None:
            recon_loss = F.mse_loss(decoded_latents, current_region_queries)
            quantization_losses["recon_loss"].append(recon_loss)


        # Final mask loss (full resolution)
        dice_loss, bce_loss, cls_loss = None, None, None
        dice_loss_normal, bce_loss_normal, cls_loss_normal = None, None, None
        dice_loss_quant, bce_loss_quant, cls_loss_quant = None, None, None
        indices_normal = None
        semantic_loss = None
        if mask_labels is not None:
            # Compute Hungarian indices using normal output; include semantic cost if semantic_labels provided
            if semantic_labels is not None and current_region_queries is not None:
                proj_queries = current_region_queries
                if self.semantic_proj is not None:
                    proj_queries = self.semantic_proj(current_region_queries)

                indices_normal, dice_loss_normal, bce_loss_normal, cls_loss_normal, semantic_loss_val = hungarian_assign_with_semantic(
                    seg_logits_normal, mask_labels, class_logits_normal, class_labels, self.num_classes, eos_coef,
                    loss_type=loss_type, current_region_queries=proj_queries, semantic_labels=semantic_labels, semantic_temperature=semantic_temperature
                )
                semantic_loss = semantic_loss_val
            else:
                indices_normal, dice_loss_normal, bce_loss_normal, cls_loss_normal = hungarian_assign(
                    seg_logits_normal, mask_labels, class_logits_normal, class_labels, self.num_classes, eos_coef, loss_type
                )
                semantic_loss = None
            # print(semantic_loss, "semantic loss!!")
            # print(dice_loss_normal, cls_loss_normal, "normal loss!")
            quantizer_info.update({
                    "hungarian_indices": indices_normal,
                })
            if seg_logits_quant is not None:
                # indices, dice_loss_quant, bce_loss_quant, cls_loss_quant = hungarian_assign(
                #     seg_logits_quant, mask_labels, class_logits_quant, class_labels, self.num_classes, eos_coef, loss_type
                # )
                # Use the same indices for quantized output
                dice_loss_quant, bce_loss_quant, cls_loss_quant = compute_losses_with_indices(
                    seg_logits_quant, mask_labels, class_logits_quant, class_labels, self.num_classes, eos_coef, loss_type, indices_normal
                )
                bce_loss = bce_loss_quant
                dice_loss = dice_loss_quant
                cls_loss = cls_loss_quant
                if quantizer_info is None:
                    quantizer_info = {}
                quantizer_info.update({
                    "hungarian_indices": indices_normal,
                    "seg_logits_quant": seg_logits_quant,
                    "class_logits_quant": class_logits_quant
                })
            else:
                dice_loss = dice_loss_normal
                bce_loss = bce_loss_normal
                cls_loss = cls_loss_normal
                # print(f"use normal losses: {dice_loss_normal}, {bce_loss_normal}, {cls_loss_normal}")

            # --- Semantic Embedding Loss (after assignment, matched pairs only) ---
            # semantic_loss already set above when semantic_labels provided

        # Aggregate quantization losses with coefficients
        
        total_vq_loss = sum(quantization_losses["vq_loss"]) if quantization_losses["vq_loss"] else None
        total_commit_loss = sum(quantization_losses["commit_loss"]) if quantization_losses["commit_loss"] else None
        total_entropy_loss = sum(quantization_losses["entropy_loss"]) if quantization_losses["entropy_loss"] else None
        total_recon_loss = sum(quantization_losses["recon_loss"]) if quantization_losses["recon_loss"] else None
        total_quantization_loss = None
        if total_vq_loss is not None and total_commit_loss is not None and total_entropy_loss is not None:
            total_quantization_loss = (
                vq_coef * total_vq_loss +
                commit_coef * total_commit_loss +
                entropy_coef * total_entropy_loss
            )
            # include reconstruction loss if present
            if total_recon_loss is not None:
                total_quantization_loss = total_quantization_loss + (recon_coef * total_recon_loss)
             # print(f"entropy loss and other: {entropy_coef * total_entropy_loss}, {vq_coef * total_vq_loss}, {commit_coef * total_commit_loss}")

        # Return hierarchical codes, masks, losses for each stage, and quantization losses
        return (
            current_region_queries, multi_scale_image_features, seg_logits_normal, class_logits_normal,
            dice_loss, bce_loss, cls_loss,
            aux_outputs if deep_supervision else None,
            hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses,
            quantization_losses, total_quantization_loss,
            dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant,
            total_recon_loss, quantizer_info,
            semantic_loss
        )

