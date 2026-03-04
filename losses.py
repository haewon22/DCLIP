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
        use_hns: bool = False,
        hns_threshold: float = 0.5,
        hns_weight: float = 2.0,
        hns_mode: str = "threshold",  
        hns_topk_ratio: float = 0.3,
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.disable_torch_grad_focal_loss = bool(disable_torch_grad_focal_loss)

        self.use_hns = bool(use_hns)
        self.hns_threshold = float(hns_threshold)
        self.hns_weight = float(hns_weight)
        self.hns_mode = str(hns_mode)
        self.hns_topk_ratio = float(hns_topk_ratio)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if logits.dim() != 3 or logits.size(1) != 2:
            raise ValueError(f"AsymmetricLoss expects logits [B,2,K], got {tuple(logits.shape)}")
        if targets.dim() != 2:
            raise ValueError(f"AsymmetricLoss expects targets [B,K], got {tuple(targets.shape)}")

        probs = self.softmax(logits)  
        p_pos = probs[:, 1, :].contiguous().reshape(-1) 
        p_neg = probs[:, 0, :].contiguous().reshape(-1)

        y = targets.reshape(-1).float().clone()

        if mask is not None:
            m = mask.reshape(-1)
            y[m == 0] = -1.0

        valid = (y != -1.0)
        if valid.sum().item() == 0:
            return logits.sum() * 0.0

        yv = y[valid]
        p_pos_v = p_pos[valid]
        p_neg_v = p_neg[valid]

        if self.clip and self.clip > 0:
            p_neg_v = (p_neg_v + self.clip).clamp(max=1.0)

        los_pos = yv * torch.log(p_pos_v.clamp(min=self.eps))
        los_neg = (1.0 - yv) * torch.log(p_neg_v.clamp(min=self.eps))
        loss = los_pos + los_neg  

        if (self.gamma_neg > 0) or (self.gamma_pos > 0):
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt = p_pos_v * yv + p_neg_v * (1.0 - yv)
                    gamma = self.gamma_pos * yv + self.gamma_neg * (1.0 - yv)
                    one_sided_w = (1.0 - pt).pow(gamma)
            else:
                pt = p_pos_v * yv + p_neg_v * (1.0 - yv)
                gamma = self.gamma_pos * yv + self.gamma_neg * (1.0 - yv)
                one_sided_w = (1.0 - pt).pow(gamma)

            loss = loss * one_sided_w

        if self.use_hns:
            w = self._compute_hns_weights(p_pos_v, yv)
            loss = loss * w

        return (-loss).mean()

    def _compute_hns_weights(self, p_pos: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = torch.ones_like(y, dtype=torch.float32)
        neg = (y < 0.5)
        if not neg.any():
            return w

        neg_probs = p_pos[neg]

        if self.hns_mode == "threshold":
            hard = neg_probs > self.hns_threshold
            neg_w = torch.ones_like(neg_probs)
            neg_w[hard] = self.hns_weight
            w[neg] = neg_w

        elif self.hns_mode == "topk":
            n = neg_probs.numel()
            k = max(1, int(n * self.hns_topk_ratio))
            _, idx = torch.topk(neg_probs, k)
            neg_w = torch.ones_like(neg_probs)
            neg_w[idx] = self.hns_weight
            w[neg] = neg_w

        elif self.hns_mode == "soft":
            neg_w = 1.0 + (self.hns_weight - 1.0) * neg_probs
            w[neg] = neg_w

        else:
            raise ValueError(f"Unknown hns_mode: {self.hns_mode}")

        return w


class MFILoss(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        lambda_mfi: float = 0.2,
        use_hns: bool = False,
        hns_topk_ratio: float = 0.5,
        clamp_min0: bool = True,

        mfi_hns_mode: str = "topk",
        beta: float = 1.0,
        eps: float = 1e-6,

        mu_mode: str = "signed",      
        scale_clamp_max: float = 0.0, 
    ):
        super().__init__()
        self.lambda_mfi = float(lambda_mfi)
        self.use_hns = bool(use_hns)
        self.hns_topk_ratio = float(hns_topk_ratio)
        self.clamp_min0 = bool(clamp_min0)

        self.mfi_hns_mode = str(mfi_hns_mode)
        self.beta = float(beta)
        self.eps = float(eps)
        self.mu_mode = str(mu_mode)
        self.scale_clamp_max = float(scale_clamp_max)

        self.bn = nn.BatchNorm1d(feature_dim, affine=False)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        if self.use_hns and self.mfi_hns_mode == "proposed":
            t = F.normalize(text_features.float(), dim=-1) 
            S = t @ t.t()                                  
            M = S.size(0)

            diag = torch.diagonal(S)
            collapse_prevention = (diag - 1.0).pow(2).sum()

            mask = ~torch.eye(M, device=S.device, dtype=torch.bool)
            off = S[mask].view(M, M - 1)

            if self.clamp_min0:
                off = off.clamp(min=0.0)

            if self.mu_mode == "signed":
                mu = off.mean(dim=1, keepdim=True)
            elif self.mu_mode == "abs":
                mu = off.abs().mean(dim=1, keepdim=True)
            elif self.mu_mode == "pos":
                mu = F.relu(off).mean(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown mu_mode: {self.mu_mode}")

            scale = self.beta / (mu + self.eps)
            if self.scale_clamp_max and self.scale_clamp_max > 0:
                scale = scale.clamp(min=-self.scale_clamp_max, max=self.scale_clamp_max)

            hns_term = (scale * off.pow(3)).sum()

            return collapse_prevention + self.lambda_mfi * hns_term

        text_bn = self.bn(text_features)
        C = (text_bn.T @ text_bn) / text_features.shape[0]

        diag = torch.diagonal(C)
        collapse_prevention = (diag - 1.0).pow(2).sum()

        D = C.size(0)
        off = C * (1.0 - torch.eye(D, device=C.device))

        if self.clamp_min0:
            off = off.clamp(min=0.0)

        if self.use_hns:
            flat = off.reshape(-1).pow(2)
            k = max(1, int(flat.numel() * self.hns_topk_ratio))
            topk_vals, _ = torch.topk(flat, k)
            reduction = topk_vals.sum()
        else:
            reduction = off.pow(2).sum()

        return collapse_prevention + self.lambda_mfi * reduction


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

        mfi_hns_mode: str = "topk",
        mfi_beta: float = 1.0,
        mfi_eps: float = 1e-6,
        mfi_mu_mode: str = "signed",
        mfi_scale_clamp_max: float = 0.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.mfi_hns_mode = str(mfi_hns_mode)

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

        self.mfi = MFILoss(
            feature_dim=feat_dim,
            lambda_mfi=lambda_mfi,
            use_hns=use_hns_mfi,
            hns_topk_ratio=mfi_topk_ratio,
            clamp_min0=mfi_clamp_min0,
            mfi_hns_mode=mfi_hns_mode,
            beta=mfi_beta,
            eps=mfi_eps,
            mu_mode=mfi_mu_mode,
            scale_clamp_max=mfi_scale_clamp_max,
        )

    def forward(self, logits, targets, text_features_2k, pos_text_features=None, mask=None):
        asl_loss = self.asl(logits, targets, mask=mask)

        if self.mfi_hns_mode == "proposed" and (pos_text_features is not None):
            mfi_inp = pos_text_features
        else:
            mfi_inp = text_features_2k

        mfi_loss = self.mfi(mfi_inp)
        total_loss = asl_loss + self.alpha * mfi_loss
        return total_loss, float(asl_loss.item()), float(mfi_loss.item())


@torch.no_grad()
def compute_inter_class_similarity(text_features: torch.Tensor) -> float:
    t = F.normalize(text_features.float(), dim=-1)
    S = t @ t.t()
    K = S.size(0)
    mask = 1.0 - torch.eye(K, device=S.device)
    off_sum = (S * mask).sum()
    return (off_sum / (K * (K - 1))).item()