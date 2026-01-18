import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def load_checkpoint(model, checkpoint_path: str, device: str = "cuda"):
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return model

    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    return model


def compute_map(y_true: np.ndarray, y_score: np.ndarray, y_mask: np.ndarray = None) -> float:
    assert y_true.shape == y_score.shape
    N, K = y_true.shape

    aps = []
    for k in range(K):
        yt = y_true[:, k]
        ys = y_score[:, k]

        valid = np.ones_like(yt, dtype=bool)

        if y_mask is not None:
            valid &= (y_mask[:, k] > 0.5)

        valid &= (yt != -1)

        yt_v = yt[valid]
        ys_v = ys[valid]

        if yt_v.size == 0 or np.sum(yt_v == 1) == 0:
            continue

        ap = average_precision_score(yt_v, ys_v)
        aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return float(np.mean(aps) * 100.0)


@torch.no_grad()
def evaluate_model(model, dataloader, device: str = "cuda") -> float:
    model.eval()

    all_labels = []
    all_scores = []
    all_masks = []

    for images, labels, mask in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(images, return_features=False)    
        probs = F.softmax(logits, dim=1)[:, 1, :]        

        all_labels.append(labels.detach().cpu().numpy())
        all_scores.append(probs.detach().cpu().numpy())
        all_masks.append(mask.detach().cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_score = np.concatenate(all_scores, axis=0)
    y_mask = np.concatenate(all_masks, axis=0)

    return compute_map(y_true, y_score, y_mask=y_mask)


@torch.no_grad()
def compute_clip_baseline_similarity(
    clip_model,
    class_names: list,
    prompt_template: str = "A photo of a {}.",
    device: str = "cuda"
) -> float:
    import clip as clip_module

    texts = [prompt_template.format(name) for name in class_names]
    tokens = clip_module.tokenize(texts).to(device)
    text_features = clip_model.encode_text(tokens)    
    text_features = F.normalize(text_features.float(), dim=-1)

    sim = text_features @ text_features.T             
    K = sim.shape[0]
    mask = 1.0 - torch.eye(K, device=sim.device)
    return ((sim * mask).sum() / (K * (K - 1))).item()


def print_training_info(cfg):
    print("=" * 60)
    print("DCLIP Training Configuration")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset}")
    print(f"CLIP Backbone: {cfg.clip_model}")
    print(f"Projected Dimension: {cfg.proj_dim}")
    print(f"Text Hidden Dimension: {cfg.text_hidden_dim}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Learning Rate: {cfg.lr}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Image Size: {cfg.image_size}")
    print("-" * 60)
    print(f"ASL: γ⁻={cfg.gamma_neg}, γ⁺={cfg.gamma_pos}, δ={cfg.asl_clip}")
    print(f"MFI: λ={cfg.mfi_lambda}, α={cfg.alpha}")
    print(f"Aggregation Scale: {cfg.aggregation_scale}")
    print("=" * 60)