from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    dataset: str = "voc2007"
    data_root: str = "./data/VOCdevkit/VOC2007"
    num_classes: int = 20

    voc_classes: List[str] = field(default_factory=lambda: [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ])

    clip_model: str = "RN101"
    proj_dim: int = 256
    text_hidden_dim: int = 384

    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 1e-4
    image_size: int = 448

    gamma_neg: float = 2.0
    gamma_pos: float = 1.0
    asl_clip: float = 0.05

    mfi_lambda: float = 0.2
    alpha: float = 7e-5

    use_hns_asl: bool = False
    asl_hns_threshold: float = 0.5
    asl_hns_weight: float = 2.0
    asl_hns_mode: str = "topk"
    asl_hns_topk_ratio: float = 0.3

    use_hns_mfi: bool = True

    mfi_topk_ratio: float = 0.1

    mfi_clamp_min0: bool = False

    mfi_hns_mode: str = "proposed"  

    mfi_beta: float = 1.0
    mfi_eps: float = 1e-6

    mfi_mu_mode: str = "signed"      
    mfi_scale_clamp_max: float = 0.0 

    positive_prompt: str = "A photo of a {}."
    negative_prompt: str = "A photo without a {}."

    use_cutout: bool = True
    cutout_n_holes: int = 1
    cutout_length: int = 112
    use_randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9

    aggregation_scale: float = 5.0

    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"

    checkpoint_dir: str = "./checkpoints_hns_proposed"
    log_interval: int = 50

def get_config(**kwargs) -> Config:
    cfg = Config()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f"Warning: Config has no attribute '{k}'")
    return cfg