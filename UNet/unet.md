# U-Net Baseline Report

**Date**: 2026-05-16  
**Model**: 2D U-Net
**Dataset**: BraTS 2024 Adult Glioma Post-Treatment (BraTS-GLI)

---

## 1. Objective

First baseline in the segmentation benchmark. 
Goal is to verify the full pipeline (data loading, preprocessing, training, inference, 
evaluation, visualization)

## 2. Dataset

- **Source**: BraTS 2024 Adult Glioma Post-Treatment training set
- **Labels (4-class)**:
  - NETC 
  - SNFH
  - ET
  - RC
- **Modalities used**: T1 / T1Gd / T2 / FLAIR
- **Data usage**: ~10% of training cases used for now
- **Train/Val split**: 1080 train cases / 270 val cases, fixed seed = 42
- **Test set**: ~10% of trainingadditional cases used for now (27 cases)

## 3. Model Architecture

2D U-Net, the foundational baseline for biomedical segmentation.

**Design Work** 

Encoder: downsamples to learn high-level features
Decoder: upsamples to get back pixel-level resolution. 
Skip connections: pass spatial detail from encoder to decoder so boundaries don't get blurred after pooling.

**What this baseline can't do well.**

- 2D only, so it can't use information from neighboring slices.
- Skip connections just concatenate features directly.
- Nothing handles class imbalance.

## 4. Training Setup

| Item | Value |
|------|-------|
| Loss | Dice loss |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 16 |
| Epochs | 20 |
| Hardware | NVIDIA RTX 5070, single GPU |

## 5. Results

Evaluated on held-out validation set (N=27 cases, fixed seed).

| Label | Dice (voxel-wise 3D) | Valid cases |
|-------|----------------------|-------------|
| NETC  | 0.6285 ± 0.2532      | 17 / 27     |
| SNFH  | 0.6799 ± 0.2829      | 27 / 27     |
| ET    | 0.7610 ± 0.2155      | 23 / 27     |
| RC    | 0.3218 ± 0.2583      | 17 / 27     |
| **Overall (mean of per-class means)** | **0.5978** | — |

**Notes on metric**: Voxel-wise Dice; cases without a given label in 
GT are excluded from that label's average.

**Note on loss function**: The original U-Net used weighted cross-entropy with a boundary weight map designed for cell segmentation. 
This benchmark uses Dice loss across all baselines.