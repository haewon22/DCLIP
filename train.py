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
from utils import evaluate_model, print_training_info


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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    alpha = (1e-5 if getattr(cfg, "mfi_hns_mode", "topk") == "proposed" else cfg.alpha)

    criterion = DCLIPLoss(
        feat_dim=cfg.proj_dim,
        alpha=cfg.alpha,
        lambda_mfi=cfg.mfi_lambda,
        gamma_neg=cfg.gamma_neg,
        gamma_pos=cfg.gamma_pos,
        clip=cfg.asl_clip,

        use_hns_asl=getattr(cfg, "use_hns_asl", False),
        asl_hns_threshold=getattr(cfg, "asl_hns_threshold", 0.5),
        asl_hns_weight=getattr(cfg, "asl_hns_weight", 2.0),
        asl_hns_mode=getattr(cfg, "asl_hns_mode", "threshold"),
        asl_hns_topk_ratio=getattr(cfg, "asl_hns_topk_ratio", 0.3),

        use_hns_mfi=getattr(cfg, "use_hns_mfi", False),
        mfi_topk_ratio=getattr(cfg, "mfi_topk_ratio", 0.3),
        mfi_clamp_min0=getattr(cfg, "mfi_clamp_min0", True),

        mfi_hns_mode=getattr(cfg, "mfi_hns_mode", "topk"),
        mfi_beta=getattr(cfg, "mfi_beta", 1.0),
        mfi_eps=getattr(cfg, "mfi_eps", 1e-6),
        mfi_mu_mode=getattr(cfg, "mfi_mu_mode", "signed"),
        mfi_scale_clamp_max=getattr(cfg, "mfi_scale_clamp_max", 0.0),
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

        t_start = time.time()

        for batch_idx, (images, labels, mask) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask   = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, text_proj_2k, pos_proj = model(images, return_features=True)

            if epoch == 1 and batch_idx == 0:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1)[:, 1, :]
                    inter_sim = compute_inter_class_similarity(pos_proj)
                    print("\n[SANITY CHECK] First batch:")
                    print(f"  logits shape: {tuple(logits.shape)} (expected: [B, 2, K])")
                    print(f"  logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                    print(f"  probs range: [{probs.min():.3f}, {probs.max():.3f}]")
                    print(f"  text_proj_2k shape: {tuple(text_proj_2k.shape)} (expected: [2K, C])")
                    print(f"  pos_proj shape: {tuple(pos_proj.shape)} (expected: [K, C])")
                    print(f"  inter-class similarity: {inter_sim:.4f}\n")

            total_loss, asl_loss, mfi_loss = criterion(
                logits, labels, text_proj_2k, pos_text_features=pos_proj, mask=mask
            )

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_asl  += asl_loss
            epoch_mfi  += mfi_loss
            num_batches += 1

        scheduler.step()

        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / num_batches
        avg_asl  = epoch_asl  / num_batches
        avg_mfi  = epoch_mfi  / num_batches

        test_map = evaluate_model(model, test_loader, device=device)

        with torch.no_grad():
            _, pos_proj = model.get_projected_text_features(normalize=True)
            current_sim = compute_inter_class_similarity(pos_proj)

        print(f"Epoch {epoch}/{cfg.epochs} ({epoch_time:.1f}s)")
        print(f"  Loss: {avg_loss:.4f} (ASL: {avg_asl:.4f}, MFI: {avg_mfi:.2f})")
        print(f"  Test mAP: {test_map:.2f}% | Inter-class sim: {current_sim:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

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