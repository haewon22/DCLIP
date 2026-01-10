import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from matplotlib.colors import LinearSegmentedColormap

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
    
    return img_pil, img_tensor


def get_similarity_maps_surgery(model, images: torch.Tensor, device: str = "cuda"):
    model.eval()
    
    B = images.shape[0]
    K = model.num_classes
    
    with torch.no_grad():
        images = images.to(device)
        
        img_local = model.clip.encode_image_local(images)  
        
        img_proj = model.image_projector(img_local.float()) 
        img_proj = F.normalize(img_proj, dim=-1)
        
        text_2k = torch.cat([model.neg_text_features, model.pos_text_features], dim=0)
        text_proj_2k = model.text_projector(text_2k) 
        
        neg_proj = F.normalize(text_proj_2k[:K], dim=-1)  
        pos_proj = F.normalize(text_proj_2k[K:], dim=-1)  
        
        sim_pos = torch.einsum('bhc,kc->bhk', img_proj, pos_proj)  
        sim_neg = torch.einsum('bhc,kc->bhk', img_proj, neg_proj)  
        
        HW = sim_pos.shape[1]
        H = W = int(HW ** 0.5)
        
        sim_maps_pos = sim_pos.permute(0, 2, 1).reshape(B, K, H, W) 
        sim_maps_neg = sim_neg.permute(0, 2, 1).reshape(B, K, H, W) 
        
        sim_pos_t = sim_pos.permute(0, 2, 1) * 5.0 
        sim_neg_t = sim_neg.permute(0, 2, 1) * 5.0
        
        q_pos = F.softmax(sim_pos_t, dim=-1)
        q_neg = F.softmax(sim_neg_t, dim=-1)
        
        p_pos = (q_pos * sim_pos_t).sum(dim=-1) * 5.0
        p_neg = (q_neg * sim_neg_t).sum(dim=-1) * 5.0
        
        logits = torch.stack([p_neg, p_pos], dim=1)
        probs = F.softmax(logits, dim=1)[:, 1, :] 
        
    return probs, sim_maps_pos, sim_maps_neg


def visualize_surgery_style(
    image_path: str,
    model,
    class_names: list,
    device: str = "cuda",
    top_k: int = 5,
    save_path: str = None,
    threshold: float = 0.0,
    use_margin: bool = True,  
):
    img_pil, img_tensor = load_image(image_path)
    img_np = np.array(img_pil.resize((448, 448)))
    
    probs, sim_maps_pos, sim_maps_neg = get_similarity_maps_surgery(model, img_tensor, device)
    
    probs = probs[0].cpu().numpy()
    sim_maps_pos = sim_maps_pos[0].cpu() 
    
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    fig, axes = plt.subplots(1, top_k + 1, figsize=(3.5 * (top_k + 1), 3.5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Image", fontsize=14, fontweight='bold')
    axes[0].axis("off")
    
    for i, idx in enumerate(top_indices):
        class_name = class_names[idx]
        prob = probs[idx]
        
        if use_margin:
            sim_class = sim_maps_pos[idx] 
            
            other_indices = [j for j in range(len(class_names)) if j != idx]
            sim_others = sim_maps_pos[other_indices]  
            sim_others_max = sim_others.max(dim=0)[0]  
            
            margin_map = sim_class - sim_others_max 
            sim_map = margin_map.numpy()
        else:
            sim_map = sim_maps_pos[idx].numpy()
        
        sim_map_up = F.interpolate(
            torch.from_numpy(sim_map).unsqueeze(0).unsqueeze(0).float(),
            size=(448, 448),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        if use_margin:
            sim_map_up = np.maximum(sim_map_up, 0)
        
        sim_min = sim_map_up.min()
        sim_max = sim_map_up.max()
        if sim_max > sim_min:
            sim_map_vis = (sim_map_up - sim_min) / (sim_max - sim_min)
        else:
            sim_map_vis = np.zeros_like(sim_map_up)
        
        axes[i + 1].imshow(img_np)
        im = axes[i + 1].imshow(sim_map_vis, alpha=0.65, cmap='jet', vmin=0, vmax=1)
        axes[i + 1].set_title(f"{class_name}\n(P={prob:.2f})", fontsize=12)
        axes[i + 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_paper_figure(
    image_path: str,
    model,
    class_names: list,
    device: str = "cuda",
    classes_to_show: list = None,
    save_path: str = None,
    use_margin: bool = True,
):

    img_pil, img_tensor = load_image(image_path)
    img_np = np.array(img_pil.resize((448, 448)))
    
    probs, sim_maps_pos, _ = get_similarity_maps_surgery(model, img_tensor, device)
    
    probs = probs[0].cpu().numpy()
    sim_maps_pos = sim_maps_pos[0].cpu()  
    
    if classes_to_show is None:
        top_indices = np.argsort(probs)[::-1][:5]
    else:
        top_indices = [class_names.index(c) for c in classes_to_show if c in class_names]
    
    n_classes = len(top_indices)
    
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(3 * (n_classes + 1), 3))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Image", fontsize=11, fontweight='bold')
    axes[0].axis("off")
    
    for i, idx in enumerate(top_indices):
        class_name = class_names[idx]
        prob = probs[idx]
        
        if use_margin:
            sim_class = sim_maps_pos[idx]
            other_indices = [j for j in range(len(class_names)) if j != idx]
            sim_others = sim_maps_pos[other_indices]
            sim_others_max = sim_others.max(dim=0)[0]
            margin_map = sim_class - sim_others_max
            sim_map = np.maximum(margin_map.numpy(), 0)
        else:
            sim_map = sim_maps_pos[idx].numpy()
        
        sim_map_up = F.interpolate(
            torch.from_numpy(sim_map).unsqueeze(0).unsqueeze(0).float(),
            size=(448, 448),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        sim_map_vis = (sim_map_up - sim_map_up.min()) / (sim_map_up.max() - sim_map_up.min() + 1e-8)
        
        axes[i + 1].imshow(img_np)
        axes[i + 1].imshow(sim_map_vis, alpha=0.6, cmap='jet')
        axes[i + 1].set_title(f"{class_name}\n({prob:.2f})", fontsize=10)
        axes[i + 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_comparison(
    image_path: str,
    model,
    class_names: list,
    device: str = "cuda",
    top_k: int = 5,
    save_path: str = None,
    use_margin: bool = True,
):
    img_pil, img_tensor = load_image(image_path)
    img_np = np.array(img_pil.resize((448, 448)))
    
    probs, sim_maps_pos, sim_maps_neg = get_similarity_maps_surgery(model, img_tensor, device)
    
    probs = probs[0].cpu().numpy()
    sim_maps_pos = sim_maps_pos[0].cpu()
    sim_maps_neg = sim_maps_neg[0].cpu()
    
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    fig, axes = plt.subplots(2, top_k + 1, figsize=(3 * (top_k + 1), 6))
    
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Image", fontsize=11, fontweight='bold')
    axes[0, 0].axis("off")
    
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title("Image", fontsize=11, fontweight='bold')
    axes[1, 0].axis("off")
    
    for i, idx in enumerate(top_indices):
        class_name = class_names[idx]
        prob = probs[idx]
        
        if use_margin:
            sim_class = sim_maps_pos[idx]
            other_indices = [j for j in range(len(class_names)) if j != idx]
            sim_others = sim_maps_pos[other_indices]
            sim_others_max = sim_others.max(dim=0)[0]
            margin_map = sim_class - sim_others_max
            sim_pos = np.maximum(margin_map.numpy(), 0)
        else:
            sim_pos = sim_maps_pos[idx].numpy()
        
        sim_pos_up = F.interpolate(
            torch.from_numpy(sim_pos).unsqueeze(0).unsqueeze(0).float(),
            size=(448, 448), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        
        sim_pos_vis = (sim_pos_up - sim_pos_up.min()) / (sim_pos_up.max() - sim_pos_up.min() + 1e-8)
        
        axes[0, i + 1].imshow(img_np)
        axes[0, i + 1].imshow(sim_pos_vis, alpha=0.6, cmap='jet')
        axes[0, i + 1].set_title(f"[+] {class_name}\n({prob:.2f})", fontsize=10)
        axes[0, i + 1].axis("off")
        
        if use_margin:
            sim_class = sim_maps_neg[idx]
            other_indices = [j for j in range(len(class_names)) if j != idx]
            sim_others = sim_maps_neg[other_indices]
            sim_others_max = sim_others.max(dim=0)[0]
            margin_map = sim_class - sim_others_max
            sim_neg = np.maximum(margin_map.numpy(), 0)
        else:
            sim_neg = sim_maps_neg[idx].numpy()
        
        sim_neg_up = F.interpolate(
            torch.from_numpy(sim_neg).unsqueeze(0).unsqueeze(0).float(),
            size=(448, 448), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        
        sim_neg_vis = (sim_neg_up - sim_neg_up.min()) / (sim_neg_up.max() - sim_neg_up.min() + 1e-8)
        
        axes[1, i + 1].imshow(img_np)
        axes[1, i + 1].imshow(sim_neg_vis, alpha=0.6, cmap='jet')
        axes[1, i + 1].set_title(f"[-] {class_name}", fontsize=10)
        axes[1, i + 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_paths = [
        "./data/VOCdevkit/VOC2007/JPEGImages/008509.jpg",
        "./data/VOCdevkit/VOC2007/JPEGImages/000129.jpg",
    ]
    
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
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model with mAP: {ckpt.get('best_map', 'N/A')}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}")
    
    model.eval()
    
    os.makedirs("./visualizations", exist_ok=True)
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        img_name = os.path.basename(img_path).split('.')[0]
        
        print(f"\nVisualizing: {img_path}")
        
        save_path = f"./visualizations/dclip_surgery_{img_name}.png"
        visualize_surgery_style(
            image_path=img_path,
            model=model,
            class_names=cfg.voc_classes,
            device=device,
            top_k=5,
            save_path=save_path,
        )
        
        save_path_cmp = f"./visualizations/dclip_comparison_{img_name}.png"
        visualize_comparison(
            image_path=img_path,
            model=model,
            class_names=cfg.voc_classes,
            device=device,
            top_k=5,
            save_path=save_path_cmp,
        )


if __name__ == "__main__":
    main()
