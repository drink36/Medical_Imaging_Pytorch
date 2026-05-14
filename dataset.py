import os
import glob
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from load import MODALITIES, load_case, normalize_nonzero


class BraTSDataset(Dataset):
    """
    2D slice dataset for BraTS segmentation.

    image: float32 tensor [4, H, W]  — 4 modalities stacked as channels
    mask:  long tensor    [H, W]     — values in {0,1,2,3,4}

    Args:
        source:     path string (scans all subdirs) OR list of case dir paths
        axis:       slicing axis — 0 sagittal, 1 coronal, 2 axial
        tumor_only: skip slices with no tumor voxels
        transform:  callable(image, mask) applied after tensor conversion
    """

    def __init__(self, source, axis=2, tumor_only=True, transform=None):
        self.axis = axis
        self.tumor_only = tumor_only
        self.transform = transform
        # volumes are cached on first access — one case stays in RAM until evicted
        self._cache: dict = {}

        if isinstance(source, (list, tuple)):
            case_dirs = sorted(source)
        else:
            case_dirs = sorted(
                d for d in glob.glob(os.path.join(source, "*")) if os.path.isdir(d)
            )
        if not case_dirs:
            raise ValueError(f"No case directories found in: {source}")

        self.samples = []  # list of (case_dir, slice_index)
        bar = tqdm.tqdm(case_dirs, desc="Loading cases", unit="case")
        for case_dir in bar:
            bar.set_postfix(case=os.path.basename(case_dir))
            try:
                _, volumes, _ = load_case(case_dir)
                self._cache[case_dir] = volumes
                n_slices = volumes["t1c"].shape[axis]
                for i in range(n_slices):
                    if tumor_only and np.take(volumes["seg"], i, axis=axis).max() == 0:
                        continue
                    self.samples.append((case_dir, i))
            except Exception as e:
                tqdm.tqdm.write(f"[BraTSDataset] Skipping {case_dir}: {e}")

        if not self.samples:
            raise ValueError("Dataset is empty after filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_dir, slice_idx = self.samples[idx]

        if case_dir not in self._cache:
            _, volumes, _ = load_case(case_dir)
            self._cache[case_dir] = volumes
        volumes = self._cache[case_dir]

        slices = [
            np.take(normalize_nonzero(volumes[mod]), slice_idx, axis=self.axis)
            for mod in MODALITIES
        ]
        image = torch.from_numpy(np.stack(slices, axis=0).astype(np.float32))  # [4, H, W]
        mask  = torch.from_numpy(
            np.take(volumes["seg"], slice_idx, axis=self.axis).astype(np.int64)
        )  # [H, W]

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


if __name__ == "__main__":
    root = "C:\\Users\\ooo91\\Desktop\\School\\Medical\\Dataset\\BraTS2024-BraTS-GLI-TrainingData\\training_data1_v2"
    ds = BraTSDataset(root, axis=2, tumor_only=True)
    print(f"Total slices: {len(ds)}")
    img, msk = ds[0]
    print(f"image: {img.shape}, dtype={img.dtype}")
    print(f"mask:  {msk.shape}, dtype={msk.dtype}, labels={msk.unique()}")
