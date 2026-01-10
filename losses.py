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
        disable_torch_grad_focal_loss: bool = True
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        x_softmax = self.softmax(x)  
        xs_pos = x_softmax[:, 1, :]  
        xs_neg = x_softmax[:, 0, :]  

        y = y.reshape(-1).clone() 
        xs_pos = xs_pos.reshape(-1)
        xs_neg = xs_neg.reshape(-1)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            y[mask_flat == 0] = -1

        valid_mask = (y != -1)
        xs_pos = xs_pos[valid_mask]
        xs_neg = xs_neg[valid_mask]
        y = y[valid_mask]
        
        num_valid = len(y)
        if num_valid == 0:
            return x.sum() * 0.0 

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            loss *= one_sided_w

        return -loss.sum() / num_valid


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
    ):
        super().__init__()
        self.alpha = alpha
        
        self.mfi = MFILoss(feature_dim=feat_dim, lambda_mfi=lambda_mfi)
        self.asl = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
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
