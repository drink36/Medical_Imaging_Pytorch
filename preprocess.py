"""
Run once to convert .nii.gz cases into pre-sliced .npy files.

Output layout:
  <out_dir>/
    BraTS-GLI-00005-100_img.npy   float32 [D, 4, H, W]  — depth first for contiguous axial slicing
    BraTS-GLI-00005-100_seg.npy   uint8   [D, H, W]      — depth first

Usage:
  python preprocess.py
"""

import os
import glob
import numpy as np
import tqdm
from load import MODALITIES, load_case, normalize_nonzero


ROOT_DIR = "C:\\Users\\ooo91\\Desktop\\School\\Medical\\Dataset\\BraTS2024-BraTS-GLI-TrainingData\\training_data1_v2"
OUT_DIR  = "C:\\Users\\ooo91\\Desktop\\School\\Medical\\Dataset\\BraTS2024-preprocessed"
DATA_RATIO = 1  # fraction of cases to preprocess for quick runs

def preprocess(root_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    case_dirs = sorted(d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d))
    # only keep a fraction of cases for quick runs
    case_dirs = case_dirs[: int(len(case_dirs) * DATA_RATIO)]
    print(f"Found {len(case_dirs)} cases in {root_dir}")

    done, skipped = 0, 0
    tqdm_bar = tqdm.tqdm(case_dirs, desc="Preprocessing cases")
    for case_dir in tqdm_bar:
        case_id = os.path.basename(case_dir)
        img_path = os.path.join(out_dir, f"{case_id}_img.npy")
        seg_path = os.path.join(out_dir, f"{case_id}_seg.npy")

        if os.path.exists(img_path) and os.path.exists(seg_path):
            print(f"  Skip (already exists): {case_id}")
            done += 1
            continue

        try:
            _, volumes, _ = load_case(case_dir)

            # Stack modalities: [4,H,W,D] → transpose to [D,4,H,W] for contiguous axial reads
            img = np.stack(
                [normalize_nonzero(volumes[m]) for m in MODALITIES], axis=0
            ).transpose(3, 0, 1, 2).astype(np.float32)  # [D, 4, H, W]

            seg = volumes["seg"].transpose(2, 0, 1).astype(np.uint8)  # [D, H, W]

            np.save(img_path, img)
            np.save(seg_path, seg)
            tqdm_bar.set_postfix(case=case_id)
            done += 1

        except Exception as e:
            skipped += 1

    print(f"\nDone: {done}  Skipped: {skipped}")


if __name__ == "__main__":
    preprocess(ROOT_DIR, OUT_DIR)
