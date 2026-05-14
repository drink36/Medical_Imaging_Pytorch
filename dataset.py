import os
import glob
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """
    2D slice dataset that loads from pre-processed .npy files.

    Expects files produced by preprocess.py:
      {npy_dir}/{case_id}_img.npy  — float32 [4, H, W, D]
      {npy_dir}/{case_id}_seg.npy  — uint8   [H, W, D]

    image: float32 tensor [4, H, W]
    mask:  long tensor    [H, W], values in {0,1,2,3,4}

    Args:
        npy_dir:    directory containing the .npy files
        case_ids:   list of case id strings to use (for train/val split).
                    If None, all cases found in npy_dir are used.
        axis:       slicing axis — 0 sagittal, 1 coronal, 2 axial
        tumor_only: skip slices with no tumor voxels
        transform:  callable(image, mask) applied after tensor conversion
    """

    def __init__(self, npy_dir, case_ids=None, axis=2, tumor_only=True, transform=None):
        self.npy_dir = npy_dir
        self.axis = axis
        self.tumor_only = tumor_only
        self.transform = transform

        if case_ids is None:
            seg_files = sorted(glob.glob(os.path.join(npy_dir, "*_seg.npy")))
            case_ids = [os.path.basename(f)[:-8] for f in seg_files]  # strip _seg.npy

        if not case_ids:
            raise ValueError(f"No .npy cases found in: {npy_dir}")

        # Load only seg to build the slice index — fast and low RAM
        self.samples = []  # list of (case_id, slice_index)
        bar = tqdm.tqdm(case_ids, desc="Indexing cases", unit="case")
        for case_id in bar:
            bar.set_postfix(case=case_id)
            try:
                seg = np.load(os.path.join(npy_dir, f"{case_id}_seg.npy"), mmap_mode="r")
                # seg is [D, H, W] — axis 0 is depth, max over H and W (axes 1,2)
                valid = np.where(seg.max(axis=(1, 2)) > 0)[0] if tumor_only else np.arange(seg.shape[0])
                for i in valid:
                    self.samples.append((case_id, int(i)))
            except Exception as e:
                tqdm.tqdm.write(f"[BraTSDataset] Skipping {case_id}: {e}")

        if not self.samples:
            raise ValueError("Dataset is empty after filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id, slice_idx = self.samples[idx]

        img = np.load(os.path.join(self.npy_dir, f"{case_id}_img.npy"), mmap_mode="r")  # [D, 4, H, W]
        seg = np.load(os.path.join(self.npy_dir, f"{case_id}_seg.npy"), mmap_mode="r")  # [D, H, W]

        # depth is axis 0 — single index gives a contiguous block from disk
        image = torch.from_numpy(img[slice_idx].astype(np.float32))   # [4, H, W]
        mask  = torch.from_numpy(seg[slice_idx].astype(np.int64))     # [H, W]

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


if __name__ == "__main__":
    npy_dir = "C:\\Users\\ooo91\\Desktop\\School\\Medical\\Dataset\\BraTS2024-preprocessed"
    ds = BraTSDataset(npy_dir, axis=2, tumor_only=True)
    print(f"Total slices: {len(ds)}")
    img, msk = ds[0]
    print(f"image: {img.shape}, dtype={img.dtype}")
    print(f"mask:  {msk.shape}, dtype={msk.dtype}, labels={msk.unique()}")
