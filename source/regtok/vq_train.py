import os
import torch

print(f"Current working directory: {os.getcwd()}")
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import math

import os
import time
import argparse
from glob import glob
from copy import deepcopy

import sys
sys.path.append("/qumulo/shared_data/aofei_summer/RegTok/source")

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from regtok.vq_model import VQ_models
from regtok.vq_loss import SegVQLoss


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    # torch.cuda.manual_seed_all(seed)
    # import numpy as np, random
    # np.random.seed(seed)
    # random.seed(seed)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
        
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # dataset = build_dataset(args, transform=transform, mask_transform=mask_transform)
    dataset = build_dataset(args, transform=None, mask_transform=mask_transform, clip_preprocess=None, use_semantic=args.use_semantic)
    data_num_classes = dataset.num_classes

    # Split dataset into train/val
    val_ratio = 0.1
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.global_seed))

    # create and load model
    # import math
    vq_model = VQ_models[args.vq_model](
        num_queries=args.num_queries,
        codebook_size=args.codebook_size,
        num_stages=args.num_stages,
        num_stacks=args.num_stacks,
        interpolate_scale_factor=math.sqrt(args.interpolate_scale_factor),
        codebook_embed_dim=args.codebook_embed_dim,
        dropout_p=args.dropout_p,
        kmeans=args.kmeans,
        num_classes=data_num_classes,
        use_quantization=args.use_quantization,
        entropy_loss_ratio=args.entropy_loss_ratio,
        finetune_codebook_only=args.finetune_codebook_only,
        upsample_mode=args.up_sample_mode,
        use_self_attn=args.use_self_attn,
        num_modalities=args.num_modalities,
        quant_use_seg=args.quant_use_seg,
        # enhanced_decoder=args.enhanced_decoder,
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)
    # for (name, param) in vq_model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable parameter: {name}, shape: {param.shape}, size: {param.numel()}, type: {param.dtype}")
    #     else:
    #         logger.info(f"Frozen parameter: {name}, shape: {param.shape}, size: {param.numel()}")

    dataset.clip_preprocess = vq_model.unimed_preprocess
    dataset.text_encoder = vq_model.text_encoder
    # print(dataset.text_encoder, "text encoder")
    if args.use_semantic:
        dataset.text_tokenizer = vq_model.text_tokenizer
    dataset.device = vq_model.device

    # DistributedSampler for train/val
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    print(dist.get_world_size(), "World size.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_collate_fn,
    )
    logger.info(f"Train set: {len(train_dataset):,} images, Val set: {len(val_dataset):,} images ({args.data_path})")

    # Use SegVQLoss for segmentation task
    vq_loss = SegVQLoss().to(device)

    if args.finetune_decoder_only: # frozen all other parameters
        if hasattr(vq_model, "image_encoder"):
            for param in vq_model.image_encoder.parameters():
                param.requires_grad_(False)

    # Print all trainable parameters
    print("Trainable parameters:")
    for name, param in vq_model.named_parameters():
        if "image_encoder" in name or "text_encoder" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
        if "quantiz" in name:
            print(name, param.requires_grad)
    if args.finetune_codebook_only:
        for name, param in vq_model.named_parameters():
            # if "quantiz" in name or "code_decoder" in name or "segmentation_cls_head" in name:
            # if "quantiz" in name or "code_decoder" in name or "to_code" in name or "segmentation_cls_head" in name or "region_perceiver" in name:
            if "quantiz" in name or "code_decoder" in name or "to_code" in name or "modality_predictor" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

            if args.kmeans:
                if "embedding" in name:
                    param.requires_grad_(False)
    for name, param in vq_model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)
    

    # No discriminator for segmentation, so skip related optimizer/scaler
    optimizer = torch.optim.Adam(
        (p for p in vq_model.parameters() if p.requires_grad),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        # weight_decay=args.weight_decay
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
        model_state = checkpoint["model"]
        # print(model_state.keys(), "keys")
        if args.finetune_decoder_only:
            # if finetuning with enhanced decoder, you would expect the old shape not match
            try:
                # if you want to continue finetune the enhanced decoder
                missing, unexpected = vq_model.load_state_dict(model_state, strict=False)
                logger.info(f"Loading Region Perceiver, Missing keys: {[i for i in missing if ('image_encoder' not in i) and ('text_encoder' not in i) and ('code_decoder' not in i)]}")
                logger.info(f"Unexpected keys: {unexpected}")
                logger.info("Load region perceiver from CKPT.")
            except:
                # if switching from small decoder to enhanced decoder, delete the old decoder keys first
                decoder_keys = [k for k in model_state.keys() if k.startswith("decoder.")]
                for k in decoder_keys:
                    del model_state[k]
                missing, unexpected = vq_model.load_state_dict(model_state, strict=False)
                logger.info(f"Missing keys: {missing}")
                logger.info(f"Unexpected keys: {unexpected}")
        else:
            vq_model.load_state_dict(model_state, strict=True)
            logger.info("Loaded model from checkpoint.")

        if args.ema:
            ema.load_state_dict(checkpoint["ema"])

        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            logger.info("Optimizer starting from scratch.")
        # try:
        #     vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        # except:
        #     logger.info("Discriminator starting from scratch.")
        
            
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
            if args.finetune_codebook_only:
                train_steps = 0
                start_epoch = 0 
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    # --- Try loading a separate quantization-specific checkpoint if provided ---
    if args.quantization_ckpt:
        try:
            qckpt = torch.load(args.quantization_ckpt, map_location="cpu", weights_only=False)
            q_model_state = qckpt.get("model", None)
            q_opt_state = qckpt.get("optimizer", None)
            if q_model_state is not None:
                try:
                    # load with strict=False so partial / quantizer-only checkpoints are accepted
                    missing, unexpected = vq_model.load_state_dict(q_model_state, strict=False)
                    logger.info(f"Loaded quantization checkpoint into model ({args.quantization_ckpt});unexpected: {unexpected}")
                    logger.info(f"Loading Region Perceiver, Missing keys: {[i for i in missing if ('image_encoder' not in i) and ('text_encoder' not in i)]}")
                except Exception as e:
                    logger.warning(f"Failed to load quantization model state from {args.quantization_ckpt}: {e}")
            if q_opt_state is not None:
                try:
                    optimizer.load_state_dict(q_opt_state)
                    logger.info(f"Loaded optimizer state from quantization checkpoint {args.quantization_ckpt}")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state from {args.quantization_ckpt}: {e}")
            # also restore ema if available
            if args.ema and "ema" in qckpt:
                try:
                    ema.load_state_dict(qckpt["ema"])
                    logger.info("Loaded EMA state from quantization checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load EMA from quantization checkpoint: {e}")
        except Exception as e:
            logger.warning(f"Unable to read quantization checkpoint {args.quantization_ckpt}: {e}")
    
    # Print all parameters and trainable parameters
    total_params = sum(p.numel() for p in vq_model.parameters())
    trainable_params = sum(p.numel() for p in vq_model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Trainable ratio: {trainable_ratio:.2%}")
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model)

    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model.train()
    if args.ema:
        ema.eval()
    # vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    # vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    log_steps = 0
    running_loss = 0
    running_seg_loss = 0
    running_bce_loss = 0
    running_dice_loss=0
    running_cls_loss = 0
    running_quantization_loss = 0
    running_distill_loss = 0
    running_semantic_loss = 0  # <-- add for semantic loss
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    if rank == 0:
        import wandb
        wandb.init(project="cloud-VQ", config=args)
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # Wrap loader with tqdm for progress bar
        epoch_iter = train_loader
        if rank == 0:
            epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch in epoch_iter:
            imgs, masks, class_labels, text_embeddings, modality_labels = batch
            imgs = imgs.to(device, non_blocking=True)
            mask_labels = [m.to(device, non_blocking=True) for m in masks]
            text_embeddings = [t.to(device, non_blocking=True) for t in text_embeddings] if text_embeddings[0] is not None else None
            modality_labels = torch.tensor(modality_labels) if modality_labels is not None else None
            if modality_labels is not None:
                modality_labels = modality_labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast(dtype=ptdtype):
            dec_mask, diff_quan, dice_loss, bce_loss, cls_loss, seg_logits, class_logits, \
            hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, \
            quantization_losses, total_quantization_loss, dice_loss_normal, dice_loss_quant, \
            cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info, semantic_loss = vq_model(imgs, do_quantize=args.use_quantization, mask_labels=mask_labels, class_labels=class_labels, semantic_labels=text_embeddings, modality_label=modality_labels, loss_type="dice_bce")
            
            loss_sum = dice_loss + bce_loss
            if cls_loss is not None:
                loss_sum += cls_loss
            if semantic_loss is not None:
                loss_sum += semantic_loss
            seg_loss = dice_loss + bce_loss
            
            if args.use_quantization:
                total_quantization_loss = total_quantization_loss * args.quantization_loss_ratio
                loss_sum += total_quantization_loss
                # print(total_quantization_loss, "quantization loss")
                if distill_loss is not None:
                    # print(f"use reconstruction loss, {distill_loss.item()}")
                    loss_sum += distill_loss
            loss_gen, loss_dict_gen = vq_loss(loss_sum)
            
            loss_gen.backward()

            if args.max_grad_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)

            optimizer.step()
            if args.ema:
                update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            running_loss += loss_gen.item()
            running_seg_loss += seg_loss.item()
            running_dice_loss += dice_loss.item()
            running_bce_loss += bce_loss.item()
            if cls_loss is not None:
                running_cls_loss += cls_loss.item()
            if args.use_quantization:
                running_quantization_loss += total_quantization_loss.item()
            running_distill_loss += distill_loss.item() if distill_loss is not None else 0
            running_semantic_loss += semantic_loss.item() if semantic_loss is not None else 0  # <-- accumulate
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_seg_loss = torch.tensor(running_seg_loss / log_steps, device=device)
                avg_dice_loss = torch.tensor(running_dice_loss / log_steps, device=device)
                avg_bce_loss = torch.tensor(running_bce_loss / log_steps, device=device)
                avg_cls_loss = torch.tensor(running_cls_loss / log_steps, device=device)
                if args.use_quantization:
                    avg_quantization_loss = torch.tensor(running_quantization_loss / log_steps, device=device)
                    avg_distill_loss = torch.tensor(running_distill_loss / log_steps, device=device)
                avg_semantic_loss = torch.tensor(running_semantic_loss / log_steps, device=device)  # <-- average
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_seg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_bce_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_dice_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_cls_loss, op=dist.ReduceOp.SUM)
                if args.use_quantization:
                    dist.all_reduce(avg_quantization_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_distill_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_semantic_loss, op=dist.ReduceOp.SUM)  # <-- reduce
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_seg_loss = avg_seg_loss.item() / dist.get_world_size()
                avg_dice_loss = avg_dice_loss.item() / dist.get_world_size()
                avg_bce_loss = avg_bce_loss.item() / dist.get_world_size()

                avg_cls_loss = avg_cls_loss.item() / dist.get_world_size()
                

                if args.use_quantization:
                    avg_quantization_loss = avg_quantization_loss.item() / dist.get_world_size()
                    avg_distill_loss = avg_distill_loss.item() / dist.get_world_size()
                else:
                    avg_quantization_loss = 0
                    avg_distill_loss = 0
                avg_semantic_loss = avg_semantic_loss.item() / dist.get_world_size()  # <-- finalize
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} (seg: {avg_seg_loss:.4f}, DICE: {avg_dice_loss:.4f}, BCE: {avg_bce_loss:.4f}, cls: {avg_cls_loss:.4f}, quant: {avg_quantization_loss:.4f}, distill: {avg_distill_loss:.4f}, semantic: {avg_semantic_loss:.4f}), Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f"Train dice_loss_normal: {dice_loss_normal}, dice_loss_quant: {dice_loss_quant}, cls_loss_normal: {cls_loss_normal}, cls_loss_quant: {cls_loss_quant}, distill_loss: {distill_loss}")

                if rank == 0:
                    wandb.log({
                        "train/total_loss": avg_loss,
                        "train/seg_loss": avg_seg_loss,
                        "train/dice_loss": avg_dice_loss,
                        "train/bce_loss": avg_bce_loss,
                        "train/cls_loss": avg_cls_loss,
                        "train/quantization_loss": avg_quantization_loss,
                        "train/distill_loss": avg_distill_loss,
                        "train/semantic_loss": avg_semantic_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/dice_loss_normal": dice_loss_normal,
                        "train/dice_loss_quant": dice_loss_quant,
                        "train/cls_loss_normal": cls_loss_normal,
                        "train/cls_loss_quant": cls_loss_quant,
                        "train/distill_loss_step": distill_loss,
                        **loss_dict_gen
                    }, step=train_steps)
 
                running_loss = 0
                running_seg_loss = 0
                running_dice_loss = 0
                running_bce_loss = 0
                running_cls_loss = 0
                running_quantization_loss = 0
                running_distill_loss = 0
                running_semantic_loss = 0  # <-- reset
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_state_full = vq_model.module._orig_mod.state_dict()
                        param_names = [name for name, p in vq_model.module._orig_mod.named_parameters() if p.requires_grad]
                    else:
                        model_state_full = vq_model.module.state_dict()
                        param_names = [name for name, p in vq_model.module.named_parameters() if p.requires_grad]
                    # Save only trainable parameters
                    model_weight = {k: v for k, v in model_state_full.items() if k in param_names}
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                    # cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    # torch.save(checkpoint, cloud_checkpoint_path)
                    # logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

        # Validation after each epoch
        running_val_dice_loss = 0
        running_val_bce_loss = 0
        running_val_distill_loss = 0
        running_val_semantic_loss = 0  # <-- add for semantic loss
        vq_model.eval()
        val_loss = 0
        val_seg_loss = 0
        val_cls_loss = 0
        val_quantization_loss = 0
        val_distill_loss = 0
        val_steps = 0
        with torch.no_grad():
            val_iter = val_loader
            if rank == 0:
                val_iter = tqdm(val_loader, desc=f"Val Epoch {epoch}", dynamic_ncols=True)
            for batch in val_iter:
                imgs, masks, class_labels, text_embeddings, modality_labels  = batch
                imgs = imgs.to(device, non_blocking=True)
                mask_labels = [m.to(device, non_blocking=True) for m in masks]
                text_embeddings = [t.to(device, non_blocking=True) for t in text_embeddings] if text_embeddings[0] is not None else None
                modality_labels = torch.tensor(modality_labels) if modality_labels is not None else None
                if modality_labels is not None:
                    modality_labels = modality_labels.to(device, non_blocking=True)
                # Unpack dice and bce loss separately
                dec_mask, diff_quan, dice_loss, bce_loss, cls_loss, seg_logits, class_logits, \
                hierarchical_codes, hierarchical_masks, hierarchical_gt_masks, hierarchical_losses, \
                quantization_losses, total_quantization_loss, dice_loss_normal, dice_loss_quant, \
                cls_loss_normal, cls_loss_quant, distill_loss, quantizer_info, semantic_loss = vq_model(
                    imgs, do_quantize=args.use_quantization, mask_labels=mask_labels, class_labels=class_labels, loss_type="dice_bce", semantic_labels=text_embeddings
                )
                seg_loss = dice_loss + bce_loss
                loss_sum = seg_loss + cls_loss
                if semantic_loss is not None:
                    loss_sum += semantic_loss
                if args.use_quantization:
                    total_quantization_loss = total_quantization_loss * args.quantization_loss_ratio
                    loss_sum += total_quantization_loss
                    if distill_loss is not None:
                        loss_sum += distill_loss
                loss_gen, _ = vq_loss(loss_sum)
                val_loss += loss_gen.item()
                val_seg_loss += seg_loss.item()
                running_val_dice_loss += dice_loss.item()
                running_val_bce_loss += bce_loss.item()
                if cls_loss is not None:
                    val_cls_loss += cls_loss.item()
                if args.use_quantization:
                    val_quantization_loss += total_quantization_loss.item()
                running_val_distill_loss += distill_loss.item() if distill_loss is not None else 0
                running_val_semantic_loss += semantic_loss.item() if semantic_loss is not None else 0
                val_distill_loss += distill_loss.item() if distill_loss is not None else 0
                val_steps += 1
        # Reduce across all processes
        val_loss_tensor = torch.tensor(val_loss / max(val_steps, 1), device=device)
        val_seg_loss_tensor = torch.tensor(val_seg_loss / max(val_steps, 1), device=device)
        val_dice_loss_tensor = torch.tensor(running_val_dice_loss / max(val_steps, 1), device=device)
        val_bce_loss_tensor = torch.tensor(running_val_bce_loss / max(val_steps, 1), device=device)
        val_cls_loss_tensor = torch.tensor(val_cls_loss / max(val_steps, 1), device=device)
        if args.use_quantization:
            val_quantization_loss_tensor = torch.tensor(val_quantization_loss / max(val_steps, 1), device=device)
        val_distill_loss_tensor = torch.tensor(val_distill_loss / max(val_steps, 1), device=device)
        val_semantic_loss_tensor = torch.tensor(running_val_semantic_loss / max(val_steps, 1), device=device)  # <-- average
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_seg_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_dice_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_bce_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_cls_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_distill_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_semantic_loss_tensor, op=dist.ReduceOp.SUM)  # <-- reduce
        val_loss_avg = val_loss_tensor.item() / dist.get_world_size()
        val_seg_loss_avg = val_seg_loss_tensor.item() / dist.get_world_size()
        val_dice_loss_avg = val_dice_loss_tensor.item() / dist.get_world_size()
        val_bce_loss_avg = val_bce_loss_tensor.item() / dist.get_world_size()
        val_cls_loss_avg = val_cls_loss_tensor.item() / dist.get_world_size()
        val_distill_loss_avg = val_distill_loss_tensor.item() / dist.get_world_size()
        val_semantic_loss_avg = val_semantic_loss_tensor.item() / dist.get_world_size()  # <-- finalize
        if args.use_quantization:
            val_quantization_loss_avg = val_quantization_loss_tensor.item() / dist.get_world_size()
        else:
            val_quantization_loss_avg = 0
        if rank == 0:
            logger.info(f"[Validation][Epoch {epoch}] Val Loss: {val_loss_avg:.4f} (seg: {val_seg_loss_avg:.4f}, DICE: {val_dice_loss_avg:.4f}, BCE: {val_bce_loss_avg:.4f}, cls: {val_cls_loss_avg:.4f}, distill: {val_distill_loss_avg:.4f}, semantic: {val_semantic_loss_avg:.4f})")
            logger.info(f"Val dice_loss_normal: {dice_loss_normal}, dice_loss_quant: {dice_loss_quant}, cls_loss_normal: {cls_loss_normal}, cls_loss_quant: {cls_loss_quant}, distill_loss: {distill_loss}")
            wandb.log({
                "val/total_loss": val_loss_avg,
                "val/seg_loss": val_seg_loss_avg,
                "val/dice_loss": val_dice_loss_avg,
                "val/bce_loss": val_bce_loss_avg,
                "val/cls_loss": val_cls_loss_avg,
                "val/quantization_loss": val_quantization_loss_avg,
                "val/distill_loss": val_distill_loss_avg,
                "val/semantic_loss": val_semantic_loss_avg,
                "val/lr": optimizer.param_groups[0]["lr"],
                "val/dice_loss_normal": dice_loss_normal,
                "val/dice_loss_quant": dice_loss_quant,
                "val/cls_loss_normal": cls_loss_normal,
                "val/cls_loss_quant": cls_loss_quant,
                "val/distill_loss_step": distill_loss,
            }, step=train_steps)
        vq_model.train()
        scheduler.step()

    # save model at the final step
    if rank == 0:
        if args.compile:
            model_state_full = vq_model.module._orig_mod.state_dict()
            param_names = [name for name, p in vq_model.module._orig_mod.named_parameters() if p.requires_grad]
        else:
            model_state_full = vq_model.module.state_dict()
            param_names = [name for name, p in vq_model.module.named_parameters() if p.requires_grad]
        # Save only trainable parameters
        model_weight = {k: v for k, v in model_state_full.items() if k in param_names}
        checkpoint = {
            "model": model_weight,
            "steps": train_steps,
            "args": args
        }
        if args.ema:
            checkpoint["ema"] = ema.state_dict()
        if not args.no_local_save:
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, cloud_checkpoint_path)
        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")

    vq_model.eval()
    logger.info("Done!")
    dist.destroy_process_group()
    wandb.finish()



def seg_collate_fn(batch):
    """
    batch: list of (image, masks)
    Returns:
        images: (B, C, H, W)
        masks: list of tensors, each (num_gt, H, W)
    """
    images = torch.cat([item[0] for item in batch], dim=0)  # each image is (1, C, H, W)
    masks = [item[1].squeeze(1) for item in batch]
    class_labels = [item[2] for item in batch]
    text_embeddings = [item[3] for item in batch]
    if len(batch[0]) == 5:
        modality_labels = [item[4] for item in batch]

    return images, masks, class_labels, text_embeddings, modality_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/path/to/your/dataset')
    parser.add_argument("--segmentation-path", type=str, default='/path/to/your/segmentation')
    parser.add_argument("--annotation-json", type=str, default='/path/to/your/annotation')
    parser.add_argument("--batch-dataset-meta-file", type=str, default=None)
    
    # parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, default='./logs/tokenflow/', help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="RegTok")
    parser.add_argument("--up-sample-mode", type=str, choices=['conv', "query"], default="conv")
    parser.add_argument("--use-self-attn", action='store_true', help="whether using self attention in region perceiver")

    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--quantization-ckpt", type=str, default=None, help="path to a checkpoint that contains trained quantizer / optimizer state (optional)")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")

    parser.add_argument("--use-quantization", action='store_true', default=False, help="whether using quantization in training, in the first stage, we do not train quantization codebook")
    parser.add_argument("--num-queries", type=int, default=16, help="number of queries for region perceiver")
    parser.add_argument("--codebook-size", type=int, default=256, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=16, help="codebook dimension for pixel vector quantization")
    parser.add_argument("--num-stages", type=int, default=3, help="region perceiver layers")
    parser.add_argument("--interpolate-scale-factor", type=float, default=4.0, help="interpolate-scale-factor for each convolution upsampling process")
    parser.add_argument("--num-stacks", type=int, default=1, help="number of stacks for region perceiver")
    # parser.add_argument("--semantic-code-dim", type=int, default=32, help="codebook dimension for semantic vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--use-semantic", action='store_true', help="use semantic labels")
    parser.add_argument("--num-modalities", type=int, default=0, help="the number of modalities for using modality-specific codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")

    parser.add_argument("--entropy-loss-ratio", type=float, default=0.1, help="BCE loss ratio in segmentation loss")
    parser.add_argument("--dice-loss-ratio", type=float, default=1.0, help="DICE loss ratio in segmentation loss")
    parser.add_argument("--quantization-loss-ratio", type=float, default=0.1, help="Quantization loss ratio in segmentation loss")
    parser.add_argument("--quant-use-seg", action='store_true', help="whether using segmentation loss to train quantization")

    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[192, 224, 256, 336], default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=512) 
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) # better change to bf16 if GPU support

    parser.add_argument("--infer_interpolate", action='store_true', help="interpolate the positional encoding for higher resolution inference")
    parser.add_argument("--kmeans", action='store_true', help="whether using kmeans for codebook initialization")
    parser.add_argument('--finetune_decoder_only', action='store_true', help='finetune decoder only')
    parser.add_argument('--finetune_codebook_only', action='store_true', help='finetune codebook only')
    args = parser.parse_args()
    main(args)