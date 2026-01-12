import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss with optional Hard Negative Sampling (HNS).
    
    HNS gives higher weight to hard negatives (negatives with high predicted probability).
    This helps the model focus on confusing/difficult negative samples.
    """
    def __init__(
        self, 
        gamma_neg: float = 2.0,  
        gamma_pos: float = 1.0, 
        clip: float = 0.05, 
        eps: float = 1e-6, 
        disable_torch_grad_focal_loss: bool = True,
        # HNS parameters
        use_hns: bool = False,
        hns_threshold: float = 0.5,      # Only consider negatives with prob > threshold as "hard"
        hns_weight: float = 2.0,         # Weight multiplier for hard negatives
        hns_mode: str = "threshold",     # "threshold", "topk", or "soft"
        hns_topk_ratio: float = 0.3,     # For topk mode: ratio of hardest negatives to upweight
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.softmax = nn.Softmax(dim=1)
        
        # HNS settings
        self.use_hns = use_hns
        self.hns_threshold = hns_threshold
        self.hns_weight = hns_weight
        self.hns_mode = hns_mode
        self.hns_topk_ratio = hns_topk_ratio

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        x_softmax = self.softmax(x)  
        xs_pos = x_softmax[:, 1, :]  # Probability of positive
        xs_neg = x_softmax[:, 0, :]  # Probability of negative

        y = y.reshape(-1).clone() 
        xs_pos_flat = xs_pos.reshape(-1)
        xs_neg_flat = xs_neg.reshape(-1)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            y[mask_flat == 0] = -1

        valid_mask = (y != -1)
        xs_pos_valid = xs_pos_flat[valid_mask]
        xs_neg_valid = xs_neg_flat[valid_mask]
        y_valid = y[valid_mask]
        
        num_valid = len(y_valid)
        if num_valid == 0:
            return x.sum() * 0.0 

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg_valid = (xs_neg_valid + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y_valid * torch.log(xs_pos_valid.clamp(min=self.eps))
        los_neg = (1 - y_valid) * torch.log(xs_neg_valid.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            pt0 = xs_pos_valid * y_valid
            pt1 = xs_neg_valid * (1 - y_valid)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_valid + self.gamma_neg * (1 - y_valid)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            loss *= one_sided_w

        # ========== Hard Negative Sampling (HNS) ==========
        if self.use_hns:
            hns_weights = self._compute_hns_weights(xs_pos_valid, y_valid)
            loss = loss * hns_weights

        return -loss.sum() / num_valid

    def _compute_hns_weights(
        self, 
        xs_pos: torch.Tensor,  # Predicted positive probability
        y: torch.Tensor,       # Ground truth labels
    ) -> torch.Tensor:
        """
        Compute HNS weights for each sample.
        
        Hard negatives = samples where y=0 but xs_pos is high (model thinks it's positive)
        """
        weights = torch.ones_like(y, dtype=torch.float32)
        
        # Only apply to negative samples
        neg_mask = (y < 0.5)  # y == 0
        
        if not neg_mask.any():
            return weights
        
        neg_probs = xs_pos[neg_mask]  # Positive prob for negative samples
        
        if self.hns_mode == "threshold":
            # Mode 1: Threshold-based
            # Negatives with prob > threshold get higher weight
            hard_neg_mask = neg_probs > self.hns_threshold
            neg_weights = torch.ones_like(neg_probs)
            neg_weights[hard_neg_mask] = self.hns_weight
            weights[neg_mask] = neg_weights
            
        elif self.hns_mode == "topk":
            # Mode 2: Top-K hardest negatives
            # Select top k% of negatives by predicted probability
            num_neg = len(neg_probs)
            k = max(1, int(num_neg * self.hns_topk_ratio))
            
            if num_neg > 0:
                _, topk_indices = torch.topk(neg_probs, k)
                neg_weights = torch.ones_like(neg_probs)
                neg_weights[topk_indices] = self.hns_weight
                weights[neg_mask] = neg_weights
                
        elif self.hns_mode == "soft":
            # Mode 3: Soft weighting based on probability
            # Higher prob → higher weight (continuous)
            # weight = 1 + (hns_weight - 1) * prob
            neg_weights = 1.0 + (self.hns_weight - 1.0) * neg_probs
            weights[neg_mask] = neg_weights
            
        else:
            raise ValueError(f"Unknown HNS mode: {self.hns_mode}")
        
        return weights


class MFILoss(nn.Module):
    def __init__(self, feature_dim: int = 256, lambda_mfi: float = 0.2):
        super().__init__()
        self.lambda_mfi = lambda_mfi
        self.bn = nn.BatchNorm1d(feature_dim, affine=False)
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        text_bn = self.bn(text_features)

        c = torch.mm(text_bn.T, text_bn)
        c = c / text_features.shape[0]
        
        feature_dim = c.shape[0]

        diagonal = torch.diag(c)
        collapse_prevention = ((diagonal - 1.0) ** 2).sum()

        off_diagonal_mask = 1.0 - torch.eye(feature_dim, device=c.device)
        mfi_reduction = (c * off_diagonal_mask).pow(2).sum()
        
        loss = collapse_prevention + self.lambda_mfi * mfi_reduction
        
        return loss
    
    @torch.no_grad()
    def inter_class_similarity(self, text_features: torch.Tensor) -> float:
        text_norm = F.normalize(text_features, dim=-1)
        sim_matrix = torch.mm(text_norm, text_norm.T)
        
        num_classes = sim_matrix.shape[0]
        mask = 1.0 - torch.eye(num_classes, device=sim_matrix.device)
        off_diagonal_sum = (sim_matrix * mask).sum()
        num_off_diagonal = num_classes * (num_classes - 1)
        
        return (off_diagonal_sum / num_off_diagonal).item()


class DCLIPLoss(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        alpha: float = 7e-5,
        lambda_mfi: float = 0.2,
        gamma_neg: float = 2.0, 
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        # HNS parameters
        use_hns: bool = False,
        hns_threshold: float = 0.5,
        hns_weight: float = 2.0,
        hns_mode: str = "threshold",
        hns_topk_ratio: float = 0.3,
    ):
        super().__init__()
        self.alpha = alpha
        self.use_hns = use_hns
        
        self.mfi = MFILoss(feature_dim=feat_dim, lambda_mfi=lambda_mfi)
        self.asl = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            use_hns=use_hns,
            hns_threshold=hns_threshold,
            hns_weight=hns_weight,
            hns_mode=hns_mode,
            hns_topk_ratio=hns_topk_ratio,
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        text_features_2k: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        asl_loss = self.asl(logits, targets, mask=mask)
        mfi_loss = self.mfi(text_features_2k)
        
        total_loss = asl_loss + self.alpha * mfi_loss
        
        return total_loss, asl_loss.item(), mfi_loss.item()


def compute_inter_class_similarity(text_features: torch.Tensor) -> float:
    with torch.no_grad():
        text_norm = F.normalize(text_features, dim=-1)
        sim_matrix = torch.mm(text_norm, text_norm.T)
        
        num_classes = sim_matrix.shape[0]
        mask = 1.0 - torch.eye(num_classes, device=sim_matrix.device)
        off_diagonal_sum = (sim_matrix * mask).sum()
        num_off_diagonal = num_classes * (num_classes - 1)
        
        return (off_diagonal_sum / num_off_diagonal).item()
