"""
DCLIP Training Script

Implements the training procedure from Section 4.2:
- SGD optimizer with lr=0.002
- Cosine annealing scheduler
- 50 epochs
- Batch size 32
- Combined loss: L_DCLIP = L_ASL + α * L_MFI
"""

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
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    # Load configuration
    cfg = Config()
    
    # Set seed
    set_seed(cfg.seed)
    
    # Device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Print config
    print_training_info(cfg)
    
    # Data loaders
    train_loader, test_loader = get_voc_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    model = create_dclip_model(
        clip_model_name=cfg.clip_model,
        class_names=cfg.voc_classes,
        num_classes=cfg.num_classes,
        proj_dim=cfg.proj_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        aggregation_scale=cfg.aggregation_scale,
        device=device,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Loss function
    criterion = DCLIPLoss(
        feat_dim=cfg.proj_dim,
        alpha=cfg.alpha,
        lambda_mfi=cfg.mfi_lambda,
        gamma_neg=cfg.gamma_neg,
        gamma_pos=cfg.gamma_pos,
        clip=cfg.asl_clip,
        # HNS options
        use_hns=cfg.use_hns,
        hns_threshold=cfg.hns_threshold,
        hns_weight=cfg.hns_weight,
        hns_mode=cfg.hns_mode,
        hns_topk_ratio=cfg.hns_topk_ratio,
    ).to(device)
    
    # Optimizer (only trainable parameters)
    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # Checkpoint directory
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
    
    # Initial inter-class similarity (before training)
    with torch.no_grad():
        _, pos_proj_init = model.get_projected_text_features(normalize=True)
        init_sim = compute_inter_class_similarity(pos_proj_init)
    print(f"\nInitial inter-class similarity: {init_sim:.4f}")
    print(f"(Target: ~0.50 for VOC, baseline CLIP: ~0.77)")
    
    # Training loop
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
            mask = mask.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            logits, text_proj_2k, pos_proj = model(images, return_features=True)
            
            # Sanity check on first batch of first epoch
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
                    print(f"  inter-class similarity: {inter_sim:.4f}")
                    print()
            
            # Compute loss
            total_loss, asl_loss, mfi_loss = criterion(
                logits, labels, text_proj_2k, mask=mask
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate
            epoch_loss += total_loss.item()
            epoch_asl += asl_loss
            epoch_mfi += mfi_loss
            num_batches += 1
        
        # Update scheduler
        scheduler.step()
        
        # Epoch statistics
        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / num_batches
        avg_asl = epoch_asl / num_batches
        avg_mfi = epoch_mfi / num_batches
        
        # Evaluation
        test_map = evaluate_model(model, test_loader, device=device)
        
        # Current inter-class similarity
        with torch.no_grad():
            _, pos_proj = model.get_projected_text_features(normalize=True)
            current_sim = compute_inter_class_similarity(pos_proj)
        
        # Print epoch results
        print(f"Epoch {epoch}/{cfg.epochs} ({epoch_time:.1f}s)")
        print(f"  Loss: {avg_loss:.4f} (ASL: {avg_asl:.4f}, MFI: {avg_mfi:.2f})")
        print(f"  Test mAP: {test_map:.2f}% | Inter-class sim: {current_sim:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
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
    
    # Final summary
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
