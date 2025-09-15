import torch
import torch.nn.functional as F
from torch import nn
import scipy

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(dim=(1,2))  # average over h and w, output shape (n,)

# batch version

# def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
#     denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss

# def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
#     hw = inputs.shape[1]
#     pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
#     neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
#     loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
#     return loss / hw

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)   # (n, hw)
    targets = targets.flatten(1) # (num_gt, hw)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.flatten(1)   # (n, hw)
    targets = targets.flatten(1) # (num_gt, hw)
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


def classification_loss(inputs: torch.Tensor, targets: torch.Tensor, num_classes, eos_coef=0.1):
    empty_weight = torch.ones(num_classes + 1, device=inputs.device)
    empty_weight[-1] = eos_coef
    loss = F.cross_entropy(inputs.transpose(1, 2), targets, weight=empty_weight)
    return loss

def hungarian_assign(
    pred_masks, gt_masks, pred_logits=None, gt_labels=None, num_classes=None, eos_coef=0.1, 
    loss_type="dice_bce", cost_class=1.0, cost_mask=1.0, cost_dice=1.0
):
    """
    pred_masks: (b, n, h, w)
    gt_masks: list of [ (num_gt, h_t, w_t) ] for each image
    pred_logits: (b, n, num_classes+1) or None
    gt_labels: list of [ (num_gt,) ] for each image
    Returns:
        indices: list of (pred_idx, gt_idx) for each image
        dice_loss: averaged matched dice loss
        bce_loss: averaged matched bce loss
        cls_loss: averaged matched classification loss
    """
    b, n, h, w = pred_masks.shape
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_cls_loss = 0.0
    total_count = 0
    indices = []
    for i in range(b):
        pm = pred_masks[i]  # (n, h, w)
        gm = gt_masks[i]    # (num_gt, h_t, w_t)
        num_gt, h_t, w_t = gm.shape
        # Print mask label stats for debugging
        # print(f"[Batch {i}] GT mask stats:")
        # for j in range(num_gt):
        #     gt_mask = gm[j]
        #     print(f"  Mask {j}: shape={gt_mask.shape}, min={gt_mask.min().item():.3f}, max={gt_mask.max().item():.3f}, mean={gt_mask.float().mean().item():.3f}, unique={torch.unique(gt_mask)}")

        # Interpolate prediction to target mask size if needed
        if (h != h_t) or (w != w_t):
            pm = F.interpolate(pm.unsqueeze(0), size=(h_t, w_t), mode="bilinear", align_corners=False).squeeze(0)
        # Pairwise mask costs
        # print(pm.size(), gm.size(), "sizes")
        dice_cost = batch_dice_loss(pm, gm).T  # (num_gt, n)
        bce_cost = batch_sigmoid_ce_loss(pm, gm).T  # (num_gt, n)
        # print(dice_cost.size(), bce_cost.size(), "cost")
        # Classification cost
        if pred_logits is not None and gt_labels is not None and num_classes is not None:
            out_prob = pred_logits[i].softmax(-1)  # (n, num_classes+1)
            tgt_ids = gt_labels[i]  # (num_gt,)
            # Get the class probability for each prediction and each GT label
            cost_class = -out_prob[:, tgt_ids]  # (n, num_gt)
            # print(out_prob.size(), tgt_ids.size(), cost_class, cost_class.size())
            cost_class = cost_class.T  # (num_gt, n)
        else:
            cost_class = 0.0
        # Total cost matrix
        # print("sizes", bce_cost.size(), dice_cost.size(), cost_class.size() if type(cost_class) is not int else "N/A")
        if loss_type == "dice_bce":
            cost_matrix = cost_mask * bce_cost + cost_dice * dice_cost + cost_class
        elif loss_type == "dice":
            cost_matrix = cost_dice * dice_cost + cost_class
        elif loss_type == "bce":
            cost_matrix = cost_mask * bce_cost + cost_class
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        indices.append((torch.as_tensor(col_ind, dtype=torch.int64), torch.as_tensor(row_ind, dtype=torch.int64)))
        # Accumulate matched dice and bce loss separately
        matched_dice_loss = dice_cost[row_ind, col_ind].sum()
        matched_bce_loss = bce_cost[row_ind, col_ind].sum()
        total_dice_loss += matched_dice_loss
        total_bce_loss += matched_bce_loss
        total_count += len(row_ind)
        # Classification loss (unchanged)
        if pred_logits is not None and gt_labels is not None and num_classes is not None:
            pred_cls = pred_logits[i][col_ind]  # (num_gt, num_classes+1)
            gt_cls = gt_labels[i][row_ind]      # (num_gt,)
            empty_weight = torch.ones(num_classes + 1, device=pm.device)
            empty_weight[-1] = eos_coef
            cls_loss = F.cross_entropy(pred_cls, gt_cls, weight=empty_weight)
            total_cls_loss += cls_loss * len(row_ind)
    dice_loss = total_dice_loss / max(total_count, 1)
    bce_loss = total_bce_loss / max(total_count, 1)
    cls_loss = total_cls_loss / max(total_count, 1) if total_cls_loss > 0 else None
    return indices, dice_loss, bce_loss, cls_loss

def compute_losses_with_indices(
    pred_masks, gt_masks, pred_logits=None, gt_labels=None, num_classes=None, eos_coef=0.1, loss_type="dice_bce", indices=None
):
    """
    Compute losses using provided Hungarian indices.
    pred_masks: (b, n, h, w)
    gt_masks: list of [ (num_gt, h_t, w_t) ] for each image
    pred_logits: (b, n, num_classes+1) or None
    gt_labels: list of [ (num_gt,) ] for each image
    indices: list of (pred_idx, gt_idx) for each image
    Returns:
        dice_loss: averaged matched dice loss
        bce_loss: averaged matched bce loss
        cls_loss: averaged matched classification loss
    """
    b, n, h, w = pred_masks.shape
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_cls_loss = 0.0
    total_count = 0
    for i in range(b):
        pm = pred_masks[i]  # (n, h, w)
        gm = gt_masks[i]    # (num_gt, h_t, w_t)
        num_gt, h_t, w_t = gm.shape
        pred_idx, gt_idx = indices[i]
        # Interpolate prediction to target mask size if needed
        if (h != h_t) or (w != w_t):
            pm = F.interpolate(pm.unsqueeze(0), size=(h_t, w_t), mode="bilinear", align_corners=False).squeeze(0)
        # Select matched predictions and GTs
        matched_pm = pm[pred_idx]      # (num_matched, h_t, w_t)
        matched_gm = gm[gt_idx]        # (num_matched, h_t, w_t)
        # Dice and BCE loss
        dice_cost = dice_loss(matched_pm, matched_gm)  # (num_matched,)
        bce_cost = bce_loss(matched_pm, matched_gm)    # (num_matched,)
        total_dice_loss += dice_cost.sum()
        total_bce_loss += bce_cost.sum()
        total_count += len(pred_idx)
        # Classification loss
        if pred_logits is not None and gt_labels is not None and num_classes is not None:
            pred_cls = pred_logits[i][pred_idx]  # (num_matched, num_classes+1)
            gt_cls = gt_labels[i][gt_idx]        # (num_matched,)
            empty_weight = torch.ones(num_classes + 1, device=pm.device)
            empty_weight[-1] = eos_coef
            cls_loss = F.cross_entropy(pred_cls, gt_cls, weight=empty_weight)
            total_cls_loss += cls_loss * len(pred_idx)
    dice_loss_avg = total_dice_loss / max(total_count, 1)
    bce_loss_avg = total_bce_loss / max(total_count, 1)
    cls_loss_avg = total_cls_loss / max(total_count, 1) if total_cls_loss > 0 else None
    return dice_loss_avg, bce_loss_avg, cls_loss_avg

def hungarian_assign_with_semantic(
    pred_masks, gt_masks, pred_logits=None, gt_labels=None, num_classes=None, eos_coef=0.1, 
    loss_type="dice_bce", cost_class=1.0, cost_mask=1.0, cost_dice=1.0,
    current_region_queries=None, semantic_labels=None, cost_semantic=1.0, semantic_temperature=1.0
):
    """
    Hungarian matching with additional semantic similarity cost.

    Args:
        pred_masks: (b, n, h, w)
        gt_masks: list of [ (num_gt, h_t, w_t) ] for each image
        pred_logits: (b, n, num_classes+1) or None
        gt_labels: list of [ (num_gt,) ] for each image
        current_region_queries: (b, n, d)
        semantic_labels: list of [ (num_gt, d) ] for each image
        cost_semantic: weight for semantic cost
        semantic_temperature: scaling for semantic similarity (higher = softer)
    Returns:
        indices: list of (pred_idx, gt_idx) for each image
        dice_loss: averaged matched dice loss
        bce_loss: averaged matched bce loss
        cls_loss: averaged matched classification loss
        semantic_loss: averaged matched semantic loss
    """
    b, n, h, w = pred_masks.shape
    total_dice_loss = 0.0
    total_bce_loss = 0.0
    total_cls_loss = 0.0
    total_semantic_loss = 0.0
    total_count = 0
    indices = []
    for i in range(b):
        pm = pred_masks[i]  # (n, h, w)
        gm = gt_masks[i]    # (num_gt, h_t, w_t)
        num_gt, h_t, w_t = gm.shape
        # Print mask label stats for debugging
        # print(f"[Batch {i}] GT mask stats:")
        # for j in range(num_gt):
        #     gt_mask = gm[j]
        #     print(f"  Mask {j}: shape={gt_mask.shape}, min={gt_mask.min().item():.3f}, max={gt_mask.max().item():.3f}, mean={gt_mask.float().mean().item():.3f}, unique={torch.unique(gt_mask)}")

        # Interpolate prediction to target mask size if needed
        if (h != h_t) or (w != w_t):
            pm = F.interpolate(pm.unsqueeze(0), size=(h_t, w_t), mode="bilinear", align_corners=False).squeeze(0)
        # Pairwise mask costs
        # print(pm.size(), gm.size(), "sizes")
        dice_cost = batch_dice_loss(pm, gm).T  # (num_gt, n)
        bce_cost = batch_sigmoid_ce_loss(pm, gm).T  # (num_gt, n)
        # print(dice_cost.size(), bce_cost.size(), "cost")
        # Classification cost
        if pred_logits is not None and gt_labels is not None and num_classes is not None:
            out_prob = pred_logits[i].softmax(-1)  # (n, num_classes+1)
            tgt_ids = gt_labels[i]  # (num_gt,)
            # Get the class probability for each prediction and each GT label
            cost_class = -out_prob[:, tgt_ids]  # (n, num_gt)
            # print(out_prob.size(), tgt_ids.size(), cost_class, cost_class.size())
            cost_class = cost_class.T  # (num_gt, n)
        else:
            cost_class = 0.0
        # --- Semantic cost ---
        if current_region_queries is not None and semantic_labels is not None:
            queries = current_region_queries[i]  # (n, d)
            semlabs = semantic_labels[i]         # (num_gt, d)
            # Normalize for cosine similarity
            queries_norm = F.normalize(queries, dim=-1)  # (n, d)
            semlabs_norm = F.normalize(semlabs, dim=-1)  # (num_gt, d)
            # Compute cosine similarity: (num_gt, n)
            # print(queries_norm.size(), semlabs_norm.size(), "semantic sizes")
            sim = torch.einsum('md,nd->mn', semlabs_norm, queries_norm) / semantic_temperature
            # Cost is negative similarity (maximize similarity = minimize cost)
            cost_sem = -sim
        else:
            cost_sem = 0.0
        # --- Total cost ---
        if loss_type == "dice_bce":
            cost_matrix = cost_mask * bce_cost + cost_dice * dice_cost + cost_class + cost_semantic * cost_sem
        elif loss_type == "dice":
            cost_matrix = cost_dice * dice_cost + cost_class + cost_semantic * cost_sem
        elif loss_type == "bce":
            cost_matrix = cost_mask * bce_cost + cost_class + cost_semantic * cost_sem
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        indices.append((torch.as_tensor(col_ind, dtype=torch.int64), torch.as_tensor(row_ind, dtype=torch.int64)))
        # Accumulate matched dice and bce loss separately
        matched_dice_loss = dice_cost[row_ind, col_ind].sum()
        matched_bce_loss = bce_cost[row_ind, col_ind].sum()
        total_dice_loss += matched_dice_loss
        total_bce_loss += matched_bce_loss
        total_count += len(row_ind)
        # Classification loss (unchanged)
        if pred_logits is not None and gt_labels is not None and num_classes is not None:
            pred_cls = pred_logits[i][col_ind]  # (num_gt, num_classes+1)
            gt_cls = gt_labels[i][row_ind]      # (num_gt,)
            empty_weight = torch.ones(num_classes + 1, device=pm.device)
            empty_weight[-1] = eos_coef
            cls_loss = F.cross_entropy(pred_cls, gt_cls, weight=empty_weight)
            total_cls_loss += cls_loss * len(row_ind)
        # Semantic loss for matched pairs
        if current_region_queries is not None and semantic_labels is not None:
            # Use cross-entropy loss: treat each matched query as a "logit" over all semantic labels
            # For each matched query, compute logits over all semantic labels and use the matched index as the target
            # matched_queries: (num_matched, d)
            # semlabs: (num_gt, d)
            matched_queries = queries[col_ind]      # (num_matched, d)
            matched_semlabs = semlabs[row_ind]      # (num_matched, d)
            matched_queries_norm = F.normalize(matched_queries, dim=-1)
            semlabs_norm = F.normalize(semlabs, dim=-1)
            # Compute logits: (num_matched, num_gt)
            logits = torch.matmul(matched_queries_norm, semlabs_norm.t()) / semantic_temperature
            # Targets: for each matched pair, the correct class is row_ind[j]
            targets = torch.arange(len(row_ind), device=logits.device)
            # Cross-entropy loss
            semantic_loss = F.cross_entropy(logits, targets)
            total_semantic_loss += semantic_loss * len(row_ind)
    dice_loss = total_dice_loss / max(total_count, 1)
    bce_loss = total_bce_loss / max(total_count, 1)
    cls_loss = total_cls_loss / max(total_count, 1) if total_cls_loss > 0 else None
    semantic_loss = total_semantic_loss / max(total_count, 1)
    return indices, dice_loss, bce_loss, cls_loss, semantic_loss
