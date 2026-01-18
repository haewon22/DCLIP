import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation

try:
    import clip as openai_clip
except Exception:
    openai_clip = None

try:
    from CLIP_Surgery import clip as CLIPS
except Exception:
    CLIPS = None


VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def minmax_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() - x.min() < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - x.min()) / (x.max() - x.min())
    return (x * 255).astype(np.uint8)


def top_n_similar(target: str, sim_df: pd.DataFrame, n: int):
    sub = sim_df[sim_df["class"] == target].sort_values("cosine_similarity", ascending=False)
    out = []
    for _, r in sub.iterrows():
        other = r["other_class"]
        if other != target:
            out.append(other)
        if len(out) >= n:
            break
    return out


def make_text_features_fn(model, model_type: str, device=None):
    device = device or get_device()
    model_type = str(model_type).lower()

    if model_type == "dclip":
        name_to_idx = {n: i for i, n in enumerate(model.class_names)}
        model.eval()

        @torch.no_grad()
        def _fn(class_names):
            feats = []
            for name in class_names:
                idx = name_to_idx[name]
                clip_feat = model.pos_text_features[idx:idx+1].to(device)
                proj = model.text_projector(clip_feat.float())
                proj = F.normalize(proj, dim=-1)
                feats.append(proj.cpu().numpy())
            return np.concatenate(feats, axis=0)

        return _fn

    if model_type == "clip":
        if openai_clip is None:
            raise ImportError("openai clip이 필요합니다.")
        model.eval()

        @torch.no_grad()
        def _fn(class_names):
            prompts = [f"A photo of a {c}." for c in class_names]
            tok = openai_clip.tokenize(prompts).to(device)
            feats = model.encode_text(tok)
            feats = F.normalize(feats.float(), dim=-1).cpu().numpy()
            return feats

        return _fn

    if model_type == "clip_surgery":
        if CLIPS is None:
            raise ImportError("CLIP_Surgery가 필요합니다.")
        model.eval()

        @torch.no_grad()
        def _fn(class_names):
            feats = CLIPS.encode_text_with_prompt_ensemble(model, class_names, device)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu().numpy()

        return _fn

    raise ValueError("Unknown model_type: %s" % model_type)


def generate_similarity_csv(text_features_fn, model_name: str, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "voc2007_class_similarity_sorted_%s.csv" % model_name.replace("/", "-"))

    if os.path.exists(out_csv):
        return out_csv

    feats = text_features_fn(VOC_CLASS_NAMES)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sim = feats @ feats.T

    records = []
    for i, cls in enumerate(VOC_CLASS_NAMES):
        sims = sim[i]
        for j in np.argsort(sims):
            records.append({
                "class": cls,
                "other_class": VOC_CLASS_NAMES[j],
                "cosine_similarity": float(sims[j]),
            })

    df = pd.DataFrame.from_records(records).sort_values(["class", "cosine_similarity"], ascending=[True, True])
    df.to_csv(out_csv, index=False)
    return out_csv


def encode_clip_local_tokens(clip_model, images: torch.Tensor) -> torch.Tensor:
    visual = clip_model.visual
    is_resnet = hasattr(visual, "layer4")

    if is_resnet:
        def stem(x):
            for conv, bn in [(visual.conv1, visual.bn1), (visual.conv2, visual.bn2), (visual.conv3, visual.bn3)]:
                x = visual.relu(bn(conv(x)))
            x = visual.avgpool(x)
            return x

        x = images.type(visual.conv1.weight.dtype)
        x = stem(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)

        B, C, H, W = x.shape
        attnpool = visual.attnpool

        tokens = x.reshape(B, C, H * W).permute(2, 0, 1)
        tokens = torch.cat([tokens.mean(dim=0, keepdim=True), tokens], dim=0) 

        pos_embed = attnpool.positional_embedding
        if pos_embed.shape[0] != H * W + 1:
            cls_pos = pos_embed[:1]
            spatial = pos_embed[1:]
            orig = int(spatial.shape[0] ** 0.5)
            spatial = spatial.permute(1, 0).reshape(1, -1, orig, orig)
            spatial = F.interpolate(spatial, size=(H, W), mode="bicubic", align_corners=False)
            spatial = spatial.reshape(-1, H * W).permute(1, 0)
            pos_embed = torch.cat([cls_pos, spatial], dim=0)

        tokens = tokens + pos_embed[:, None, :].to(tokens.dtype)
        tokens = F.linear(tokens, attnpool.v_proj.weight, attnpool.v_proj.bias)
        tokens = F.linear(tokens, attnpool.c_proj.weight, attnpool.c_proj.bias)

        tokens = tokens[1:].permute(1, 0, 2)
        return tokens

    x = images.type(visual.conv1.weight.dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    cls_tok = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_tok, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    x = visual.ln_post(x)
    tokens = x[:, 1:, :]
    if visual.proj is not None:
        tokens = tokens @ visual.proj
    return tokens

def make_segment_fn(model, model_type: str, image_size=448, device=None):
    device = device or get_device()
    model_type = str(model_type).lower()

    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    if model_type == "dclip":
        preprocess = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize])
        name_to_idx = {n: i for i, n in enumerate(model.class_names)}
        model.eval()

        @torch.no_grad()
        def _segment(image_pil, target_class, negative_classes, threshold=0.5, temp=5.0):
            classes = [target_class] + list(negative_classes)
            img = preprocess(image_pil.convert("RGB")).unsqueeze(0).to(device)

            img_local = model.clip.encode_image_local(img)
            img_proj = model.image_projector(img_local.float())
            img_proj = F.normalize(img_proj, dim=-1)

            hw = img_proj.shape[1]
            h = int(np.sqrt(hw))
            w = hw // h
            if h * w != hw:
                h = w = int(hw ** 0.5)

            W0, H0 = image_pil.size 
            masks = {}

            for cname in classes:
                idx = name_to_idx[cname]
                pos_clip = model.pos_text_features[idx:idx+1].to(device)
                neg_clip = model.neg_text_features[idx:idx+1].to(device)

                pos = F.normalize(model.text_projector(pos_clip.float()), dim=-1)
                neg = F.normalize(model.text_projector(neg_clip.float()), dim=-1)

                sim_pos = (img_proj @ pos.t()).squeeze(-1) * temp 
                sim_neg = (img_proj @ neg.t()).squeeze(-1) * temp

                prob_pos = torch.softmax(torch.stack([sim_neg, sim_pos], dim=-1), dim=-1)[..., 1]  # (1,HW)

                m = prob_pos[0].reshape(h, w).detach().cpu().numpy()
                m = (m * 255).astype(np.uint8)
                m = np.array(Image.fromarray(m).resize((W0, H0), resample=Image.BICUBIC))
                masks[cname] = (m > int(255 * threshold))

            return masks

        return _segment

    if model_type == "clip":
        if openai_clip is None:
            raise ImportError("openai clip이 필요합니다.")
        preprocess = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize])
        model.eval()

        @torch.no_grad()
        def _segment(image_pil, target_class, negative_classes, threshold=0.5):
            texts = [target_class] + list(negative_classes)
            prompts = [f"A photo of a {t}." for t in texts]

            img = preprocess(image_pil.convert("RGB")).unsqueeze(0).to(device)
            tokens = encode_clip_local_tokens(model, img)
            tokens = F.normalize(tokens.float(), dim=-1)

            tok = openai_clip.tokenize(prompts).to(device)
            text_feat = model.encode_text(tok)
            text_feat = F.normalize(text_feat.float(), dim=-1)

            sim = torch.matmul(tokens, text_feat.t())[0].cpu().numpy() 
            hw = sim.shape[0]
            h = int(np.sqrt(hw))
            w = hw // h
            if h * w != hw:
                h = w = int(hw ** 0.5)

            W0, H0 = image_pil.size
            masks = {}
            for j, name in enumerate(texts):
                m = sim[:, j].reshape(h, w)
                m = minmax_uint8(m)
                m = np.array(Image.fromarray(m).resize((W0, H0), resample=Image.BICUBIC))
                masks[name] = (m > int(255 * threshold))
            return masks

        return _segment

    if model_type == "clip_surgery":
        if CLIPS is None:
            raise ImportError("CLIP_Surgery가 필요합니다.")

        preprocess = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            normalize,
        ])
        model.eval()

        @torch.no_grad()
        def _segment(image_pil, target_class, negative_classes, threshold=0.5):
            import cv2
            cv2_img = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            texts = [target_class] + list(negative_classes)

            img = preprocess(image_pil).unsqueeze(0).to(device)
            image_features = model.encode_image(img)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = CLIPS.encode_text_with_prompt_ensemble(model, texts, device)
            similarity = CLIPS.clip_feature_surgery(image_features, text_features)
            similarity_map = CLIPS.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])  # (1,H,W,N)

            masks = {}
            for n, name in enumerate(texts):
                m = similarity_map[0, :, :, n].detach().cpu().numpy()
                m = minmax_uint8(m)
                masks[name] = (m > int(255 * threshold))
            return masks

        return _segment

    raise ValueError("Unknown model_type: %s" % model_type)


def run_unmixrate_voc2007(
    model,
    model_type: str,
    model_name: str,
    voc_root="datasets",
    year="2007",
    image_set="test",
    threshold=0.5,
    n_similar_negatives=5,
    max_images=200,
    out_dir="outputs",
    image_size=448,
):
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)

    text_fn = make_text_features_fn(model, model_type=model_type, device=device)
    sim_csv = generate_similarity_csv(text_fn, model_name=model_name, out_dir=out_dir)
    sim_df = pd.read_csv(sim_csv)

    seg_fn = make_segment_fn(model, model_type=model_type, image_size=image_size, device=device)

    voc = VOCSegmentation(root=voc_root, year=year, image_set=image_set, download=False)
    total = len(voc) if max_images is None else min(len(voc), int(max_images))

    results = []
    for i in range(total):
        image_pil, gt_mask_pil = voc[i]
        gt = np.array(gt_mask_pil, dtype=np.uint8)

        present = np.unique(gt).tolist()
        present = [p for p in present if p not in (0, 255)]
        if len(present) == 0:
            continue

        for cls_id in present:
            target_class = VOC_CLASS_NAMES[int(cls_id) - 1]
            negs = top_n_similar(target_class, sim_df, n_similar_negatives)
            if not negs:
                continue

            gt_mask = (gt == cls_id)
            masks = seg_fn(image_pil, target_class, negs, threshold=threshold)

            iou = {}
            for name, pmask in masks.items():
                pmask = pmask.astype(bool)
                inter = np.logical_and(gt_mask, pmask).sum()
                union = np.logical_or(gt_mask, pmask).sum()
                iou[name] = float(inter / union) if union > 0 else 0.0

            neg_ious = [iou[n] for n in negs if n in iou]
            neg_term = 1.0 - float(np.mean(neg_ious)) if len(neg_ious) else 0.0

            tgt = iou.get(target_class, 0.0)
            if tgt > 0 and neg_term > 0:
                unmix = 2.0 / ((1.0 / tgt) + (1.0 / neg_term))
            else:
                unmix = 0.0

            results.append({
                "image_id": os.path.basename(voc.images[i]).split(".")[0],
                "target_class": target_class,
                "target_iou": tgt,
                "neg_term": neg_term,
                "unmixrate": unmix,
                "negs": ",".join(negs),
            })

    df = pd.DataFrame(results)
    avg = float(df["unmixrate"].mean()) if len(df) else 0.0

    out_csv = os.path.join(
        out_dir,
        "unmixrate_%s_%s_%s_thr%s_N%s_avg%.4f.csv" % (
            year,
            image_set,
            model_name.replace("/", "-"),
            threshold,
            n_similar_negatives,
            avg
        )
    )
    df.to_csv(out_csv, index=False)
    return df, avg, out_csv
