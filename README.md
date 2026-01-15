# Unofficial DCLIP (Paper Reproduction)

This repository is an **unofficial implementation** of **DCLIP** from the paper:

- **Paper**: *DCLIP* — https://arxiv.org/pdf/2502.02977  
- **Status**: **No official public code** was available at the time of implementation, so this repo reproduces the method based on the paper description and ablations.

---

## What is DCLIP?

DCLIP is a CLIP-based multi-label recognition approach that learns **decoupled text embeddings** (positive/negative prompts) and trains lightweight **projection heads** to better align:

- **local image tokens** (from CLIP vision backbone)
- with **projected text features** (pos/neg per class)

This allows DCLIP to improve **classification** (e.g., VOC2007 mAP) while also supporting **localization-style maps** via token–text similarity.

---

## Visualization (Surgery-style)

This repo produces **CLIP Feature Surgery–style** activation maps in the **projected space (256-d)**, which matches the paper’s visualization direction and is also used for UnmixRate mask generation in this implementation.

![DCLIP unified visualization](visualizations/dclip_visualization_unified.png)

---

## What is UnmixRate?
> Note: UnmixRate is a metric proposed in this repository (not from the original DCLIP paper).

**UnmixRate** is a **leakage-aware** segmentation-style score intended to measure not only *how well the target class is localized*, but also *how well confusing “similar” classes are suppressed*.

For a target class:

- **target_iou**: IoU between predicted mask and GT mask for the target class  
- Select **N most similar negative classes** (by cosine similarity of **text features**)  
- **leakage**: mean IoU between GT target mask and masks produced by those similar negatives  
- **neg_term** = 1 − leakage  
- **UnmixRate** = harmonic mean of `target_iou` and `neg_term`

**Interpretation**
- High **mIoU** alone can still be “bad” if the model also fires strongly for confusing negatives (high leakage).
- UnmixRate penalizes that behavior, so it’s more sensitive to **class confusion / leakage** than target IoU alone.
- In that sense, UnmixRate can be viewed as a supplementary metric that **complements mIoU** by explicitly capturing confusion/leakage from similar negatives.


This repo also includes a **synthetic demo cell** that intentionally constructs cases where:
- mIoU is high but leakage is large → UnmixRate drops
- mIoU is slightly lower but leakage is near-zero → UnmixRate improves

---

## Result highlight (this implementation)

- In my experiments, **my DCLIP implementation not only achieved strong mAP**, but also scored **high on UnmixRate**, indicating improved separation from confusing negatives under the leakage-aware metric.
---

## Quick Start

### 1) Dataset layout

This repo assumes VOC is placed under:

```
./data/VOCdevkit/VOC2007/...
```

### 2) Train (example)

```bash
python train.py
```

### 3) Evaluate

- **mAP**: VOC2007 classification (test split)
- **UnmixRate**: VOC2007 segmentation (commonly on `test`; see notes below)

If you use the notebook, it runs end-to-end:
- model loading
- mAP evaluation
- UnmixRate evaluation
- supplementary UnmixRate vs mIoU demo

---

## Notes on “test vs val” for UnmixRate

If DCLIP training used a validation split for model selection, then reporting UnmixRate on **VOCSegmentation test** is the cleaner choice for final reporting (to avoid tuning on the same split).  
In this repo, the UnmixRate runner supports `image_set="val"` or `"test"` via `VOCSegmentation(..., image_set=...)`.

---

## Disclaimer

- This is **not** the official codebase.
- Some implementation choices are “paper-faithful best-effort”, especially where the paper does not specify exact engineering details.
- If you spot a discrepancy vs. the paper, PRs/issues are welcome.

---

## Citation

If you use this repo, please cite the original paper:

- https://arxiv.org/pdf/2502.02977
