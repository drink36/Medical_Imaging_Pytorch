"""
Run once to convert .nii.gz cases into pre-sliced .npy files.

Output layout:
  <out_dir>/
    BraTS-GLI-00005-100_img.npy   float32 [4, H, W, D]  — 4 modalities stacked
    BraTS-GLI-00005-100_seg.npy   uint8   [H, W, D]      — segmentation labels

Usage:
  python preprocess.py
"""

import os
import glob
import numpy as np

from load import MODALITIES, load_case, normalize_nonzero

ROOT_DIR = "C:\\Users\\ooo91\\Desktop\\School\\Medical\\Dataset\\BraTS2024-BraTS-GLI-TrainingData\\training_data1_v2"
OUT_DIR  = "C:\\Users\\ooo91\\Desktop\\School\\Medical\\Dataset\\BraTS2024-preprocessed"


def preprocess(root_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    case_dirs = sorted(d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d))
    print(f"Found {len(case_dirs)} cases in {root_dir}")

    done, skipped = 0, 0
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        img_path = os.path.join(out_dir, f"{case_id}_img.npy")
        seg_path = os.path.join(out_dir, f"{case_id}_seg.npy")

        if os.path.exists(img_path) and os.path.exists(seg_path):
            print(f"  Skip (already exists): {case_id}")
            done += 1
            continue

        try:
            _, volumes, _ = load_case(case_dir)

            # Stack modalities and normalize — shape [4, H, W, D]
            img = np.stack(
                [normalize_nonzero(volumes[m]) for m in MODALITIES], axis=0
            ).astype(np.float32)

            seg = volumes["seg"].astype(np.uint8)   # [H, W, D]

            np.save(img_path, img)
            np.save(seg_path, seg)
            print(f"  OK: {case_id}  img={img.shape}  labels={np.unique(seg)}")
            done += 1

        except Exception as e:
            print(f"  SKIP {case_id}: {e}")
            skipped += 1

    print(f"\nDone: {done}  Skipped: {skipped}")


if __name__ == "__main__":
    preprocess(ROOT_DIR, OUT_DIR)
