"""
DCLIP Configuration (Paper-Faithful)

Paper: Efficiently Disentangling CLIP for Multi-Object Perception

Implementation Details (Sec. 4.2):
- epochs: 50
- batch_size: 32
- optimizer: SGD, lr=0.002, cosine annealing
- image_size: 448
- ASL: gamma_neg=2, gamma_pos=1, delta=0.05
- MFI: lambda=0.2, alpha=7e-5 (for COCO-14 pretrain)
- Projectors: image [clip_dim → 256], text [clip_dim → 384 → 256]
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ===================
    # Dataset
    # ===================
    dataset: str = "voc2007"
    data_root: str = "./data/VOCdevkit/VOC2007"
    num_classes: int = 20

    voc_classes: List[str] = field(default_factory=lambda: [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ])

    # ===================
    # Model (Sec. 4.2)
    # ===================
    clip_model: str = "RN101"       # Paper uses RN101 for main results
    proj_dim: int = 256             # Projected dimension
    text_hidden_dim: int = 384      # Text projector hidden dim

    # ===================
    # Training (Sec. 4.2)
    # ===================
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.002               # Initial learning rate
    momentum: float = 0.9
    weight_decay: float = 1e-4
    image_size: int = 448

    # ===================
    # ASL Loss (Sec. 4.2, Eq. 5)
    # DCLIP paper: γ- = 2, γ+ = 1, δ = 0.05
    # ===================
    gamma_neg: float = 2.0          # DCLIP paper value
    gamma_pos: float = 1.0
    asl_clip: float = 0.05          # delta

    # ===================
    # MFI Loss (Sec. 4.2, Eq. 2)
    # λ = 0.2, α = 7e-5
    # ===================
    mfi_lambda: float = 0.2
    alpha: float = 7e-5
    
    # # ===================
    # # Hard Negative Sampling (HNS)
    # # Upweight hard negatives (negatives with high positive probability)
    # # ===================
    # use_hns: bool = False           # Enable/disable HNS
    # hns_threshold: float = 0.5      # For threshold mode: negatives with prob > this are "hard"
    # hns_weight: float = 2.0         # Weight multiplier for hard negatives
    # hns_mode: str = "topk"     # "threshold", "topk", or "soft"
    # hns_topk_ratio: float = 0.5     # For topk mode: ratio of hardest negatives

    use_hns_asl: bool = False          # ASL HNS 켜기
    asl_hns_threshold: float = 0.5
    asl_hns_weight: float = 2.0
    asl_hns_mode: str = "topk"        # "threshold" or "topk"
    asl_hns_topk_ratio: float = 0.3   # 하위 30%의 어려운 샘플 집중

    # ===================
    # MFI HNS (Feature Decorrelation) - NEW
    # ===================
    use_hns_mfi: bool = True          # MFI HNS 켜기 (추천)
    mfi_topk_ratio: float = 0.1       # 상위 10%의 높은 상관관계(Hard Pairs)만 집중 공격
    mfi_clamp_min0: bool = True       # 양의 상관관계만 줄임 (음수는 허용)
    # ===================
    # Prompts (Sec. 3.2)
    # ===================
    positive_prompt: str = "A photo of a {}."
    negative_prompt: str = "A photo without a {}."

    # ===================
    # Augmentation (Sec. 4.2)
    # ===================
    use_cutout: bool = True
    cutout_n_holes: int = 1
    cutout_length: int = 112
    use_randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9

    # ===================
    # Aggregation (Algorithm 2)
    # ===================
    aggregation_scale: float = 5.0  # "* 5" in Algorithm 2

    # ===================
    # Misc
    # ===================
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50


def get_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    cfg = Config()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f"Warning: Config has no attribute '{k}'")
    return cfg