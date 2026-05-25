import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from dataset.build import build_dataset
from regtok.vq_model import VQ_models

def dice_score(pred, gt, eps=1e-6):
    # pred, gt: (H, W), binary masks
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + eps)

def iou_score(pred, gt, eps=1e-6):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + eps)

def precision_score(pred, gt, eps=1e-6):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    return tp / (tp + fp + eps)

def recall_score(pred, gt, eps=1e-6):
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    return tp / (tp + fn + eps)

def f1_score(pred, gt, eps=1e-6):
    prec = precision_score(pred, gt, eps)
    rec = recall_score(pred, gt, eps)
    return 2 * prec * rec / (prec + rec + eps)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = build_dataset(
        type("Args", (), {
            "dataset": "biomed_seg",
            "data_path": args.data_path,
            "segmentation_path": args.mask_path,
            "annotation_json": args.annotation_json,
            "image_size": args.image_size
        })(),
        transform=None,
        mask_transform=mask_transform,
        clip_preprocess=None
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=lambda batch: (
        torch.cat([item[0] for item in batch], dim=0),
        [item[1].squeeze(1) for item in batch],
        [item[2] for item in batch]
    ))

    vq_model = VQ_models["RegTok"](
        codebook_size=0,
        num_stages=3,
        codebook_embed_dim=64,
        dropout_p=0.0,
        kmeans=False,
        upsample_mode="query",
        use_quantization=False,
        num_classes=getattr(dataset, "num_classes", 2)
    )
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = vq_model.load_state_dict(ckpt["model"], strict=False)
    # print(ckpt["model"].keys())
    print(f"Loading Region Perceiver, Missing keys: {[i for i in missing if ('image_encoder' not in i)]}")
    print(f"Unexpected keys: {unexpected}")
    print("Load region perceiver from CKPT.")
    vq_model = vq_model.to(device)
    vq_model.eval()

    dataset.clip_preprocess = vq_model.unimed_preprocess

    results_path = os.path.join(os.path.dirname(args.ckpt_path), "segmentation_abd_eval_results.txt")
    # Save intermediate results to a separate file
    intermediate_path = os.path.join(os.path.dirname(args.ckpt_path), "segmentation_abd_eval_intermediate.npy")

    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    with torch.no_grad():
        for imgs, masks, class_labels in tqdm(loader):
            imgs = imgs.to(device)
            masks = [i.to(device) for i in masks]
            outputs = vq_model(
                imgs, do_quantize=False, mask_labels=masks, class_labels=[c.to(device) for c in class_labels], loss_type="dice_bce"
            )
            # if len(outputs) == 7:
            dec, diff, dice_loss, bce_loss, cls_loss, seg_logits, class_logits, hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, quantization_losses, total_quantization_loss, dice_loss_normal, dice_loss_quant, cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info = outputs
            # dec_mask, _, _, _, _, seg_logits, _ = outputs
            # else:
            #     dec_mask, _, _, _ = outputs
            #     seg_logits = dec_mask

            imgs_np = imgs.cpu().numpy()
            for b in range(imgs_np.shape[0]):
                gt_masks = masks[b].cpu().numpy()  # (num_gt, H, W)
                pred_masks_all = seg_logits[b]     # (N, H_pred, W_pred)
                num_gt, H_gt, W_gt = gt_masks.shape
                N, H_pred, W_pred = pred_masks_all.shape
                # Interpolate prediction to GT mask size if needed
                if (H_pred != H_gt) or (W_pred != W_gt):
                    pred_masks_all = torch.nn.functional.interpolate(
                        pred_masks_all.unsqueeze(0), size=(H_gt, W_gt), mode="bilinear", align_corners=False
                    ).squeeze(0)
                pred_masks_all = pred_masks_all.cpu().numpy()  # (N, H_gt, W_gt)
                # For each GT mask, find best matching pred mask (greedy)
                for g in range(num_gt):
                    gt_mask = gt_masks[g]
                    metric_per_pred = []
                    for p in range(pred_masks_all.shape[0]):
                        pred_mask = (pred_masks_all[p] > 0).astype(np.uint8)
                        dice = dice_score(pred_mask, gt_mask)
                        iou = iou_score(pred_mask, gt_mask)
                        prec = precision_score(pred_mask, gt_mask)
                        rec = recall_score(pred_mask, gt_mask)
                        f1 = f1_score(pred_mask, gt_mask)
                        metric_per_pred.append((dice, iou, prec, rec, f1))
                    if metric_per_pred:
                        best_metrics = max(metric_per_pred, key=lambda x: x[0])  # select by best dice
                        dice_scores.append(best_metrics[0])
                        iou_scores.append(best_metrics[1])
                        precision_scores.append(best_metrics[2])
                        recall_scores.append(best_metrics[3])
                        f1_scores.append(best_metrics[4])
            # print(f"Mean DICE score step: {np.mean(dice_scores):.4f}")
            # Save intermediate results after each batch
            np.save(intermediate_path, {
                "dice": np.array(dice_scores),
                "iou": np.array(iou_scores),
                "precision": np.array(precision_scores),
                "recall": np.array(recall_scores),
                "f1": np.array(f1_scores),
            })
    print(f"Mean DICE score: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU score: {np.mean(iou_scores):.4f}")
    print(f"Mean Precision: {np.mean(precision_scores):.4f}")
    print(f"Mean Recall: {np.mean(recall_scores):.4f}")
    print(f"Mean F1-score: {np.mean(f1_scores):.4f}")
    print(f"Median DICE score: {np.median(dice_scores):.4f}")
    print(f"Sample count: {len(dice_scores)}")

    with open(results_path, "w") as f:
        f.write(f"Mean DICE score: {np.mean(dice_scores):.4f}\n")
        f.write(f"Mean IoU score: {np.mean(iou_scores):.4f}\n")
        f.write(f"Mean Precision: {np.mean(precision_scores):.4f}\n")
        f.write(f"Mean Recall: {np.mean(recall_scores):.4f}\n")
        f.write(f"Mean F1-score: {np.mean(f1_scores):.4f}\n")
        f.write(f"Median DICE score: {np.median(dice_scores):.4f}\n")
        f.write(f"Sample count: {len(dice_scores)}\n")
    print(f"Intermediate results saved to: {intermediate_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--annotation_json", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    main(args)
