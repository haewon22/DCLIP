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
    # Model
    # ===================
    clip_model: str = "RN101"
    proj_dim: int = 256      
    text_hidden_dim: int = 384

    # ===================
    # Training
    # ===================
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.002  
    momentum: float = 0.9
    weight_decay: float = 1e-4
    image_size: int = 448

    # ===================
    # ASL Loss
    # paper: γ- = 2, γ+ = 1, δ = 0.05
    # ===================
    gamma_neg: float = 2.0  
    gamma_pos: float = 1.0
    asl_clip: float = 0.05  

    # ===================
    # MFI Loss
    # λ = 0.2, α = 7e-5
    # ===================
    mfi_lambda: float = 0.2
    alpha: float = 7e-5
    
    use_hns_asl: bool = False       
    asl_hns_threshold: float = 0.5
    asl_hns_weight: float = 2.0
    asl_hns_mode: str = "topk"      
    asl_hns_topk_ratio: float = 0.3 

    # ===================
    # MFI HNS 
    # ===================
    use_hns_mfi: bool = True      
    mfi_topk_ratio: float = 0.1   
    mfi_clamp_min0: bool = True   
    # ===================
    # Prompts
    # ===================
    positive_prompt: str = "A photo of a {}."
    negative_prompt: str = "A photo without a {}."

    # ===================
    # Augmentation 
    # ===================
    use_cutout: bool = True
    cutout_n_holes: int = 1
    cutout_length: int = 112
    use_randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9

    # ===================
    # Aggregation 
    # ===================
    aggregation_scale: float = 5.0  

    # ===================
    # Misc
    # ===================
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50


def get_config(**kwargs) -> Config:
    cfg = Config()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f"Warning: Config has no attribute '{k}'")
    return cfg