import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from config import Config
from models import create_dclip_model


def load_image(image_path: str, image_size: int = 448):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)
    img_vis = np.array(img_pil.resize((image_size, image_size)))
    return img_vis, img_tensor


def _infer_hw(n_tokens: int, image_size: int):
    s = int(round(math.sqrt(n_tokens)))
    if s * s == n_tokens:
        return s, s
    g = image_size // 32
    if g * g == n_tokens:
        return g, g
    return s, max(1, n_tokens // max(1, s))


def _minmax_norm(x: torch.Tensor, eps: float = 1e-8):

    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _upsample_to(x: torch.Tensor, out_hw: int):
    return F.interpolate(x, size=(out_hw, out_hw), mode="bilinear", align_corners=False)


def _overlay(ax, img_np, heat_np, title: str = "", alpha: float = 0.6):
    ax.imshow(img_np)
    ax.imshow(heat_np, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    if title:
        ax.set_title(title, fontsize=11)
    ax.axis("off")

@torch.no_grad()
def compute_surgery_map(
    model,
    images: torch.Tensor,
    image_size: int = 448,
    tau_surgery: float = 2.0,
):
    device = next(model.parameters()).device
    images = images.to(device)

    logits = model(images, return_features=False)
    probs = F.softmax(logits, dim=1)[:, 1, :]

    img_local = model.clip.encode_image_local(images)
    img_proj = model.image_projector(img_local.float())
    img_proj = F.normalize(img_proj, dim=-1)

    B, HW, C = img_proj.shape
    Ht, Wt = _infer_hw(HW, image_size)

    K = model.num_classes
    pos_text = model.pos_text_features
    pos_proj = model.text_projector(pos_text).float()
    pos_proj = F.normalize(pos_proj, dim=-1)

    Fi = img_proj
    Ft = pos_proj

    Fc = F.normalize(Fi.mean(dim=1), dim=-1)

    s = torch.softmax(tau_surgery * (Fc @ Ft.t()), dim=-1)
    w = s / (s.mean(dim=-1, keepdim=True) + 1e-8)

    Fm = Fi.unsqueeze(2) * Ft.view(1, 1, K, C)
    Fr = (Fm * w.view(B, 1, K, 1)).mean(dim=2)
    S = (Fm - Fr.unsqueeze(2)).sum(dim=-1)

    surg_maps = S.permute(0, 2, 1).reshape(B, K, Ht, Wt)
    surg_maps = F.relu(surg_maps)

    surg_maps_u = _upsample_to(surg_maps, image_size)
    surg_maps_u = _minmax_norm(surg_maps_u)
    # surg_maps_u = surg_maps_u * probs.unsqueeze(-1).unsqueeze(-1)

    return probs, surg_maps_u


# def visualize_surgery(
#     image_path: str,
#     model,
#     class_names: list,
#     device: str = "cuda",
#     top_k: int = 5,
#     save_path: str = None,
#     image_size: int = 448,
#     tau_surgery: float = 2.0,
# ):
#     img_np, img_tensor = load_image(image_path, image_size=image_size)

#     probs, surg_u = compute_surgery_map(
#         model,
#         img_tensor.to(device),
#         image_size=image_size,
#         tau_surgery=tau_surgery,
#     )

#     probs_np = probs[0].detach().cpu().numpy()
#     top_idx = np.argsort(probs_np)[::-1][:top_k]

#     fig, axes = plt.subplots(top_k, 2, figsize=(10, 3.6 * top_k))
#     if top_k == 1:
#         axes = np.expand_dims(axes, axis=0)

#     axes[0, 0].set_title("Image", fontsize=13, fontweight="bold")
#     axes[0, 1].set_title("top-k visualization", fontsize=13, fontweight="bold")

#     for r, k in enumerate(top_idx):
#         cname = class_names[k]
#         p = float(probs_np[k])

#         axes[r, 0].imshow(img_np)
#         axes[r, 0].set_title(f"{cname} (P={p:.3f})", fontsize=12, fontweight="bold")
#         axes[r, 0].axis("off")

#         sur_map = surg_u[0, k].detach().cpu().numpy()
#         _overlay(axes[r, 1], img_np, sur_map, title="")

#     plt.tight_layout()

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
#         print(f"Saved: {save_path}")
#     else:
#         plt.show()

#     plt.close()

# --- add these defaults near visualize_surgery args (or keep as local constants inside) ---
# (no comments requested, so just code)

def visualize_surgery(
    image_path: str,
    model,
    class_names: list,
    device: str = "cuda",
    top_k: int = 5,
    save_path: str = None,
    image_size: int = 448,
    tau_surgery: float = 2.0,
    conf_thr: float = 0.30,
    base_alpha: float = 0.60,
    sharp: float = 12.0,
):
    img_np, img_tensor = load_image(image_path, image_size=image_size)

    probs, surg_u = compute_surgery_map(
        model,
        img_tensor.to(device),
        image_size=image_size,
        tau_surgery=tau_surgery,
    )

    probs_np = probs[0].detach().cpu().numpy()

    sorted_idx = np.argsort(probs_np)[::-1]
    top_idx = [i for i in sorted_idx if probs_np[i] >= conf_thr][:top_k]
    if len(top_idx) == 0:
        top_idx = sorted_idx[:min(top_k, len(sorted_idx))].tolist()

    n_rows = len(top_idx)

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.6 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    axes[0, 0].set_title("Image", fontsize=13, fontweight="bold")
    axes[0, 1].set_title(f"Top-k Visualization", fontsize=13, fontweight="bold")

    def alpha_from_p(p):
        gate = 1.0 / (1.0 + np.exp(-sharp * (p - conf_thr)))
        return float(base_alpha * gate)

    for r, k in enumerate(top_idx):
        cname = class_names[k]
        p = float(probs_np[k])

        axes[r, 0].imshow(img_np)
        axes[r, 0].set_title(f"{cname} (P={p:.3f})", fontsize=12, fontweight="bold")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(img_np)
        axes[r, 1].axis("off")

        if p >= conf_thr:
            sur_map = surg_u[0, k].detach().cpu().numpy()
            axes[r, 1].imshow(sur_map, cmap="jet", alpha=alpha_from_p(p), vmin=0, vmax=1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_surgery_unified(
    image_paths: list,
    model,
    class_names: list,
    device: str = "cuda",
    top_k: int = 2,
    save_path: str = None,
    image_size: int = 448,
    tau_surgery: float = 2.0,
    conf_thr: float = 0.30,
    base_alpha: float = 0.60,
    sharp: float = 12.0,
):
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if len(valid_paths) == 0:
        return

    def alpha_from_p(p):
        gate = 1.0 / (1.0 + np.exp(-sharp * (p - conf_thr)))
        return float(base_alpha * gate)

    n = len(valid_paths)
    fig, axes = plt.subplots(n, 1 + top_k, figsize=(5.2 * (1 + top_k), 3.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    axes[0, 0].set_title("Image", fontsize=13, fontweight="bold")
    for c in range(top_k):
        axes[0, 1 + c].set_title(f"Top-{c+1}", fontsize=13, fontweight="bold")

    for r, img_path in enumerate(valid_paths):
        img_np, img_tensor = load_image(img_path, image_size=image_size)
        probs, surg_u = compute_surgery_map(
            model,
            img_tensor.to(device),
            image_size=image_size,
            tau_surgery=tau_surgery,
        )
        probs_np = probs[0].detach().cpu().numpy()
        sorted_idx = np.argsort(probs_np)[::-1]

        picked = [i for i in sorted_idx if probs_np[i] >= conf_thr]
        if len(picked) < top_k:
            for i in sorted_idx:
                if i not in picked:
                    picked.append(i)
                if len(picked) >= top_k:
                    break
        picked = picked[:top_k]

        axes[r, 0].imshow(img_np)
        axes[r, 0].set_title(os.path.basename(img_path), fontsize=12, fontweight="bold")
        axes[r, 0].axis("off")

        for j, k in enumerate(picked):
            cname = class_names[k]
            p = float(probs_np[k])
            axes[r, 1 + j].imshow(img_np)
            axes[r, 1 + j].axis("off")
            axes[r, 1 + j].set_title(f"{cname} (P={p:.3f})", fontsize=11, fontweight="bold")
            if p >= conf_thr:
                sur_map = surg_u[0, k].detach().cpu().numpy()
                axes[r, 1 + j].imshow(sur_map, cmap="jet", alpha=alpha_from_p(p), vmin=0, vmax=1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model = create_dclip_model(
        clip_model_name=cfg.clip_model,
        class_names=cfg.voc_classes,
        num_classes=cfg.num_classes,
        proj_dim=cfg.proj_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        aggregation_scale=cfg.aggregation_scale,
        device=device,
    )

    ckpt_path = "./checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model | best mAP: {ckpt.get('best_map', 'N/A')}")
    else:
        print(f"[WARN] checkpoint not found: {ckpt_path}")

    model.eval()

    image_paths = [
        "./data/VOCdevkit/VOC2007/JPEGImages/008509.jpg",
        "./data/VOCdevkit/VOC2007/JPEGImages/000129.jpg",
        "./data/VOCdevkit/VOC2007/JPEGImages/004287.jpg",
        "./data/VOCdevkit/VOC2007/JPEGImages/006308.jpg",
    ]

    out_dir = "./visualizations"
    os.makedirs(out_dir, exist_ok=True)

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[SKIP] image not found: {img_path}")
            continue

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(out_dir, f"dclip_visualization_{img_name}.png")

        visualize_surgery(
            image_path=img_path,
            model=model,
            class_names=cfg.voc_classes,
            device=device,
            top_k=5,
            save_path=save_path,
            image_size=cfg.image_size,
            tau_surgery=2.0,
        )

    unified_path = os.path.join(out_dir, "dclip_visualization_unified.png")
    visualize_surgery_unified(
        image_paths=image_paths,
        model=model,
        class_names=cfg.voc_classes,
        device=device,
        top_k=2,
        save_path=unified_path,
        image_size=cfg.image_size,
        tau_surgery=2.0,
    )


if __name__ == "__main__":
    main()
