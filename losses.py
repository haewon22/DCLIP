import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(
        self, 
        gamma_neg: float = 2.0,  
        gamma_pos: float = 1.0, 
        clip: float = 0.05, 
        eps: float = 1e-6, 
        disable_torch_grad_focal_loss: bool = True,
        # HNS parameters
        use_hns: bool = False,
        hns_threshold: float = 0.5,
        hns_weight: float = 2.0,
        hns_mode: str = "threshold",
        hns_topk_ratio: float = 0.3,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.softmax = nn.Softmax(dim=1)
        
        self.use_hns = use_hns
        self.hns_threshold = hns_threshold
        self.hns_weight = hns_weight
        self.hns_mode = hns_mode
        self.hns_topk_ratio = hns_topk_ratio

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        x_softmax = self.softmax(x)  
        xs_pos = x_softmax[:, 1, :]
        xs_neg = x_softmax[:, 0, :]

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

        if self.clip is not None and self.clip > 0:
            xs_neg_valid = (xs_neg_valid + self.clip).clamp(max=1)

        los_pos = y_valid * torch.log(xs_pos_valid.clamp(min=self.eps))
        los_neg = (1 - y_valid) * torch.log(xs_neg_valid.clamp(min=self.eps))
        loss = los_pos + los_neg

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

        if self.use_hns:
            hns_weights = self._compute_hns_weights(xs_pos_valid, y_valid)
            loss = loss * hns_weights

        return -loss.sum() / num_valid

    def _compute_hns_weights(
        self, 
        xs_pos: torch.Tensor,
        y: torch.Tensor, 
    ) -> torch.Tensor:
        weights = torch.ones_like(y, dtype=torch.float32)
        
        neg_mask = (y < 0.5) 
        
        if not neg_mask.any():
            return weights
        
        neg_probs = xs_pos[neg_mask]
        
        if self.hns_mode == "threshold":
            hard_neg_mask = neg_probs > self.hns_threshold
            neg_weights = torch.ones_like(neg_probs)
            neg_weights[hard_neg_mask] = self.hns_weight
            weights[neg_mask] = neg_weights
            
        elif self.hns_mode == "topk":
            num_neg = len(neg_probs)
            k = max(1, int(num_neg * self.hns_topk_ratio))
            
            if num_neg > 0:
                _, topk_indices = torch.topk(neg_probs, k)
                neg_weights = torch.ones_like(neg_probs)
                neg_weights[topk_indices] = self.hns_weight
                weights[neg_mask] = neg_weights
                
        elif self.hns_mode == "soft":
            neg_weights = 1.0 + (self.hns_weight - 1.0) * neg_probs
            weights[neg_mask] = neg_weights
            
        else:
            raise ValueError(f"Unknown HNS mode: {self.hns_mode}")
        
        return weights


class MFILoss(nn.Module):
    def __init__(
        self, 
        feature_dim: int = 256, 
        lambda_mfi: float = 0.2,
        use_hns: bool = False,
        hns_topk_ratio: float = 0.5,  
        clamp_min0: bool = True       
    ):
        super().__init__()
        self.lambda_mfi = lambda_mfi
        self.bn = nn.BatchNorm1d(feature_dim, affine=False)
        
        self.use_hns = use_hns
        self.hns_topk_ratio = hns_topk_ratio
        self.clamp_min0 = clamp_min0
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        text_bn = self.bn(text_features)
        c = torch.mm(text_bn.T, text_bn)
        c = c / text_features.shape[0]
        
        feature_dim = c.shape[0]
        
        diagonal = torch.diag(c)
        collapse_prevention = ((diagonal - 1.0) ** 2).sum()

        off_diagonal_mask = 1.0 - torch.eye(feature_dim, device=c.device)
        
        off_diag_c = c * off_diagonal_mask
        
        if self.clamp_min0:
            off_diag_c = off_diag_c.clamp(min=0)

        if self.use_hns:
            flat_c = off_diag_c.view(-1)
            flat_c_sq = flat_c.pow(2)
            
            num_elements = flat_c.numel()
            k = max(1, int(num_elements * self.hns_topk_ratio))
            
            topk_values, _ = torch.topk(flat_c_sq, k)
            
            mfi_reduction = topk_values.sum()

        else:
            mfi_reduction = off_diag_c.pow(2).sum()
        
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
        
        use_hns_asl: bool = False,
        asl_hns_threshold: float = 0.5,
        asl_hns_weight: float = 2.0,
        asl_hns_mode: str = "threshold",
        asl_hns_topk_ratio: float = 0.3,

        use_hns_mfi: bool = False,
        mfi_topk_ratio: float = 0.3,
        mfi_clamp_min0: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        
        self.mfi = MFILoss(
            feature_dim=feat_dim, 
            lambda_mfi=lambda_mfi,
            use_hns=use_hns_mfi,
            hns_topk_ratio=mfi_topk_ratio,
            clamp_min0=mfi_clamp_min0
        )
        
        self.asl = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            use_hns=use_hns_asl,
            hns_threshold=asl_hns_threshold,
            hns_weight=asl_hns_weight,
            hns_mode=asl_hns_mode,
            hns_topk_ratio=asl_hns_topk_ratio,
        )
    
    def forward(self, logits, targets, text_features_2k, mask=None):
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