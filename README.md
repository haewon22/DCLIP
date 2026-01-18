# Unofficial DCLIP (Efficiently Disentangling CLIP) + HNS Extensions

Unofficial PyTorch implementation of **DCLIP** from the paper **“Efficiently Disentangling CLIP for Multi-Object Perception”** with two extra ideas:
- **HNS (Hard Negative Sampling)** options for **ASL** and **MFI** losses  
- **UnmixRate** evaluation utilities (VOCSegmentation-based)

> Note: This repo is **unofficial** (the authors did not release a public reference implementation at the time of writing).

## Paper
- arXiv: **Efficiently Disentangling CLIP for Multi-Object Perception** (arXiv:2502.02977)  
  https://arxiv.org/abs/2502.02977

## What’s inside
- `models.py` : DCLIP (frozen CLIP + image/text projectors + local aggregation)
- `losses.py` : ASL + MFI + (optional) HNS for each
- `dataset.py` : VOC2007 multi-label loader (with label-mask handling)
- `train.py` : VOC2007 training + mAP eval + inter-class similarity tracking
- `visualization.py` : CLIP Surgery-style localization visualization (projected space)
- `UnmixRate.ipynb` : UnmixRate + mAP comparisons (DCLIP / CLIP / CLIP Surgery)


(Optional) If you want CLIP Surgery baselines in `UnmixRate.ipynb`, install your CLIP Surgery package/repo as needed.

## Data
Default expects **VOC2007** at:
```
./data/VOCdevkit/VOC2007
```
Make sure `ImageSets/Main/*.txt` and `JPEGImages/*.jpg` exist.

## Train (VOC2007)
```bash
python train.py
```
Key configs are in `config.py`:
- `clip_model` (e.g., RN101), `proj_dim`, `text_hidden_dim`
- ASL: `gamma_neg`, `gamma_pos`, `asl_clip`
- MFI: `mfi_lambda`, `alpha`
- HNS toggles:
  - `use_hns_asl`, `asl_hns_mode` (`threshold|topk|soft`), etc.
  - `use_hns_mfi`, `mfi_topk_ratio`, `mfi_clamp_min0`

Checkpoints:
- Saved to `./checkpoints/best_model.pth`

## Visualize (localization maps)
```bash
python visualization.py
```
Outputs go to `./visualizations/`.

## UnmixRate
Open and run:
- `UnmixRate.ipynb`

It computes:
- **mAP** on VOC2007 multi-label classification
- **UnmixRate** on VOCSegmentation (harmonic mean of target IoU and “non-leakage” vs similar negatives)

## Citation
If you use this code, please cite the original paper:
```bibtex
@article{dclip2025,
  title={Efficiently Disentangling CLIP for Multi-Object Perception},
  journal={arXiv preprint arXiv:2502.02977},
  year={2025}
}
```
