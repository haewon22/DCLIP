import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from dataset import get_voc_dataloaders
from models import create_dclip_model
from losses import DCLIPLoss, compute_inter_class_similarity
from utils import evaluate_model, print_training_info, compute_clip_baseline_similarity


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    cfg = Config()

    set_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print_training_info(cfg)

    train_loader, test_loader = get_voc_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = create_dclip_model(
        clip_model_name=cfg.clip_model,
        class_names=cfg.voc_classes,
        num_classes=cfg.num_classes,
        proj_dim=cfg.proj_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        aggregation_scale=cfg.aggregation_scale,
        device=device,
    )

    with torch.no_grad():
        clip_sim = compute_clip_baseline_similarity(
            model.clip.clip_model, cfg.voc_classes, device=device
        )
    print(f"CLIP baseline inter-class sim (pre-proj): {clip_sim:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    criterion = DCLIPLoss(
        feat_dim=cfg.proj_dim,
        alpha=cfg.alpha,
        lambda_mfi=cfg.mfi_lambda,
        gamma_neg=cfg.gamma_neg,
        gamma_pos=cfg.gamma_pos,
        clip=cfg.asl_clip,
    ).to(device)

    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")

    with torch.no_grad():
        _, pos_proj_init = model.get_projected_text_features(normalize=True)
        init_sim = compute_inter_class_similarity(pos_proj_init)
    print(f"\nInitial inter-class similarity: {init_sim:.4f}")
    print(f"(Target: ~0.50 for VOC, baseline CLIP: ~0.77)")

    best_map = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_asl = 0.0
        epoch_mfi = 0.0
        num_batches = 0

        dbg_pos_margin_sum = 0.0
        dbg_pos_margin_cnt = 0

        dbg_fp_prob_sum = 0.0
        dbg_fp_prob_cnt = 0

        dbg_mfi_offdiag_sum = 0.0
        dbg_mfi_offdiag_cnt = 0

        t_start = time.time()

        for batch_idx, (images, labels, mask) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, text_proj_2k, pos_proj = model(images, return_features=True)

            with torch.no_grad():
                pos_prob = F.softmax(logits, dim=1)[:, 1, :]   
                margin = logits[:, 1, :] - logits[:, 0, :]    

                valid = (mask > 0)

                pos_sel = valid & (labels > 0.5)
                if pos_sel.any():
                    dbg_pos_margin_sum += margin[pos_sel].sum().item()
                    dbg_pos_margin_cnt += pos_sel.sum().item()

                neg_sel = valid & (labels < 0.5)
                if neg_sel.any():
                    dbg_fp_prob_sum += pos_prob[neg_sel].sum().item()
                    dbg_fp_prob_cnt += neg_sel.sum().item()

                bn = criterion.mfi.bn
                was_training = bn.training
                bn.eval() 
                t_bn = bn(text_proj_2k)
                if was_training:
                    bn.train()

                c = (t_bn.T @ t_bn) / text_proj_2k.shape[0]
                C = c.shape[0]
                off_mask = 1.0 - torch.eye(C, device=c.device)
                offdiag_mean = (c * off_mask).pow(2).sum() / (C * (C - 1))

                dbg_mfi_offdiag_sum += offdiag_mean.item()
                dbg_mfi_offdiag_cnt += 1

            if epoch == 1 and batch_idx == 0:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1)[:, 1, :]
                    inter_sim = compute_inter_class_similarity(pos_proj)

                    local_feat = model.clip.encode_image_local(images)
                    print("\n[SANITY CHECK] First batch:")
                    print(f"  images shape: {tuple(images.shape)}")
                    print(f"  local_feat shape: {tuple(local_feat.shape)} (expected: [B, N, 512], e.g., N=196 for 448)")
                    print(f"  logits shape: {tuple(logits.shape)} (expected: [B, 2, K])")
                    print(f"  logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                    print(f"  probs range: [{probs.min():.3f}, {probs.max():.3f}]")
                    print(f"  text_proj_2k shape: {tuple(text_proj_2k.shape)} (expected: [2K, C])")
                    print(f"  pos_proj shape: {tuple(pos_proj.shape)} (expected: [K, C])")
                    print(f"  inter-class similarity: {inter_sim:.4f}")
                    print()

            total_loss, asl_loss, mfi_loss = criterion(
                logits, labels, text_proj_2k, mask=mask
            )

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_asl += asl_loss
            epoch_mfi += mfi_loss
            num_batches += 1

        scheduler.step()

        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / num_batches
        avg_asl = epoch_asl / num_batches
        avg_mfi = epoch_mfi / num_batches

        test_map = evaluate_model(model, test_loader, device=device)

        with torch.no_grad():
            _, pos_proj = model.get_projected_text_features(normalize=True)
            current_sim = compute_inter_class_similarity(pos_proj)

        print(f"Epoch {epoch}/{cfg.epochs} ({epoch_time:.1f}s)")
        print(f"  Loss: {avg_loss:.4f} (ASL: {avg_asl:.4f}, MFI: {avg_mfi:.2f})")
        print(f"  Test mAP: {test_map:.2f}% | Inter-class sim: {current_sim:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        pos_margin_mean = dbg_pos_margin_sum / max(1.0, dbg_pos_margin_cnt)
        fp_prob_mean = dbg_fp_prob_sum / max(1.0, dbg_fp_prob_cnt)
        mfi_off_mean = dbg_mfi_offdiag_sum / max(1, dbg_mfi_offdiag_cnt)

        print(f"  [DBG] pos margin (pos): {pos_margin_mean:.4f}")
        print(f"  [DBG] pos prob (neg):  {fp_prob_mean:.4f}")
        print(f"  [DBG] MFI offdiag^2:    {mfi_off_mean:.6f}")

        if test_map > best_map:
            best_map = test_map
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_map": best_map,
                "inter_class_sim": current_sim,
            }, best_path)
            print(f"  ★ New best! Saved to {best_path}")

        print()

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best mAP: {best_map:.2f}%")
    print(f"Paper target: ~95.4% mAP for VOC2007 with RN101")
    print(f"Final inter-class similarity: {current_sim:.4f}")
    print(f"Paper target: ~0.50 (down from CLIP's ~0.77)")
    print("=" * 60)


if __name__ == "__main__":
    train()
