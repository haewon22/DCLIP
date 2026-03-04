# Unofficial DCLIP (Paper Reproduction)

This repository is an **unofficial implementation** of **DCLIP** from the paper:

- **Paper (arXiv PDF)**: https://arxiv.org/pdf/2502.02977  
- **Paper (arXiv page)**: https://arxiv.org/abs/2502.02977  
- **Status**: **No official public code** was available at the time of implementation, so this repo reproduces the method based on the paper description and ablations.

---

## What is DCLIP?

DCLIP is a CLIP-based multi-label recognition approach that learns **decoupled text embeddings** (positive/negative prompts) and trains lightweight **projection heads** to better align:

- **local image tokens** (from CLIP vision backbone)
- with **projected text features** (pos/neg per class)

This allows DCLIP to improve **classification** (e.g., VOC2007 mAP) while also supporting **localization-style maps** via token–text similarity.

---

## Visualization (Surgery-style)

This repo produces CLIP Feature Surgery–style activation maps in the projected space (256-d), which matches the paper’s visualization direction and is also used for UnmixRate mask generation in this implementation.


![DCLIP unified visualization](visualizations/dclip_visualization_unified.png)

---

## What is UnmixRate?
> Note: **UnmixRate is a metric proposed in this repository** (not from the original DCLIP paper).

**UnmixRate** is a **leakage-aware** segmentation-style score that tries to measure two things at once:

1) **Target localization quality** (does the target class mask match GT?)  
2) **Confusion/leakage suppression** (do “similar” negative classes also fire on the same GT region?)

For a target class:

- **target_iou**: IoU between predicted mask and GT mask for the target class  
- Choose **N most similar negative classes** using cosine similarity of **text features**  
- For each similar negative class, produce its mask and compute IoU against the **GT target region**  
- **leakage**: mean IoU of those negative masks vs GT target  
- **neg_term** = 1 − leakage  
- **UnmixRate** = harmonic mean of `target_iou` and `neg_term`

**Interpretation**
- High **target IoU** can still be misleading if the model also activates strongly for confusing negatives (high leakage).
- UnmixRate penalizes that behavior, so it is more sensitive to **class confusion / leakage** than target IoU alone.
- Practically, it can be used as a supplementary metric that **complements mIoU/IoU** by explicitly capturing “unmixing” ability.

This repo also includes a small **synthetic demo cell** to illustrate cases where:
- target IoU is high but leakage is large → UnmixRate drops
- target IoU is slightly lower but leakage is near-zero → UnmixRate improves

---

## What is HNS?
> Note: **HNS is an extension explored in this repository** (not from the original DCLIP paper).

This repo contains two usages that historically shared the name “HNS”:

- **ASL-HNS (Hard Negative reweighting)**: among negative labels, up-weight negatives that the model currently predicts as high probability (likely false positives).  
  Supported modes:
  - `threshold`: up-weight negatives with `p(pos) > threshold`
  - `topk`: up-weight top-k fraction of negatives by `p(pos)`
  - `soft`: smoothly weight by probability

- **MFI-HNS**:
  - `topk` mode: focuses the MFI off-diagonal penalty on the largest offenders (top-k fraction) rather than summing all off-diagonals equally.
  - `proposed` mode: implements the paper-PDF-inspired **cubic similarity** regularizer (s^3 with normalization) to emphasize highly entangled class pairs.

All HNS knobs are exposed in `config.py`.

---

## Result highlight (this implementation)

- In my experiments, this DCLIP implementation achieved **strong VOC2007 mAP** and supports leakage-aware analysis via **UnmixRate / TargetIoU / Leakage**.

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
- **UnmixRate**: VOCSegmentation-based evaluation (commonly on `val` or `test`)

Recommended (terminal-friendly) evaluation script:

```bash
python run_results.py \
  --voc-root ./data \
  --voc2007-root ./data/VOCdevkit/VOC2007 \
  --dclip-ckpt checkpoints/best_model.pth \
  --dclip-hns-ckpt checkpoints_hns_proposed/best_model.pth \
  --thr 0.5 --topn 5 --unmix-split val \
  --max-images 200 \
  --out-dir outputs
```

---

## Disclaimer

- This is **not** the official codebase.
- Some implementation choices are “paper-faithful best-effort”, especially where the paper does not specify exact engineering details.
- If you spot a discrepancy vs. the paper, PRs/issues are welcome.

---

## Citation

If you use this repo, please cite the original paper:

- https://arxiv.org/abs/2502.02977
