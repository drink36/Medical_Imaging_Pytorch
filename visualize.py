"""
Visualize inference results.

Usage:
  python visualize.py --case_dir path/to/BraTS-GLI-00005-100
                      --pred     path/to/predictions/BraTS-GLI-00005-100-pred.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from load import load_nifti, get_case_paths, normalize_nonzero, get_best_slices, LABELSTONAME
from infer import dice_3d

MODALITY = "t1c"
ALPHA    = 0.5


# [background, NETC,    SNFH,    ET,      RC]
SEG_COLORS = ["none", "red", "green", "blue", "yellow"]
CMAP_SEG   = mcolors.ListedColormap(SEG_COLORS)
SEG_NORM   = mcolors.BoundaryNorm(boundaries=[0, 1, 2, 3, 4, 5], ncolors=5)


def load_pred(pred_path: str) -> np.ndarray:
    return nib.load(pred_path).get_fdata().astype(np.uint8)


def show(case_dir: str, pred_path: str, modality: str = "t1c", gt_path: str = None, save_path: str = None) -> None:
    _, paths      = get_case_paths(case_dir)
    img, _, _, _  = load_nifti(paths[modality])
    img           = normalize_nonzero(img)
    pred          = load_pred(pred_path)

    # resolve GT: explicit path > same case dir > None
    import os
    _gt_path = gt_path or (paths["seg"] if os.path.exists(paths["seg"]) else None)
    gt       = load_nifti(_gt_path)[0].astype(np.uint8) if _gt_path else None

    x, y, z = get_best_slices(pred)

    views = [
        (img[:, :, z], pred[:, :, z], gt[:, :, z] if gt is not None else None, f"Axial z={z}"),
        (img[:, y, :], pred[:, y, :], gt[:, y, :] if gt is not None else None, f"Coronal y={y}"),
        (img[x, :, :], pred[x, :, :], gt[x, :, :] if gt is not None else None, f"Sagittal x={x}"),
    ]

    n_rows = 3 if gt is not None else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    fig.suptitle(f"Prediction — {paths[modality].split(chr(92))[-2]}  [{modality}]", fontsize=13)

    row_labels = ["Image + Prediction", "Prediction only"]
    if gt is not None:
        row_labels.append("Image + Ground Truth")

    ROW_TITLES = {
        0: "Prediction",
        1: "Prediction only (no MRI)",
        2: "Ground Truth",
    }

    for col, (img2d, pred2d, gt2d, title) in enumerate(views):
        axes[0, col].imshow(img2d.T, cmap="gray", origin="lower")
        axes[0, col].imshow(np.ma.masked_where(pred2d.T == 0, pred2d.T),
                            cmap=CMAP_SEG, norm=SEG_NORM, alpha=ALPHA, origin="lower")
        axes[0, col].set_title(title)
        axes[0, col].axis("off")

        axes[1, col].imshow(pred2d.T, cmap=CMAP_SEG, norm=SEG_NORM, origin="lower")
        axes[1, col].set_title(title)
        axes[1, col].axis("off")

        if gt is not None:
            axes[2, col].imshow(img2d.T, cmap="gray", origin="lower")
            axes[2, col].imshow(np.ma.masked_where(gt2d.T == 0, gt2d.T),
                                cmap=CMAP_SEG, norm=SEG_NORM, alpha=ALPHA, origin="lower")
            axes[2, col].set_title(title)
            axes[2, col].axis("off")

    # set_ylabel is hidden by axis("off") — use fig.text instead
    for row, label in ROW_TITLES.items():
        if row < n_rows:
            y = 1 - (row + 0.5) / n_rows
            fig.text(0.01, y, label, va="center", ha="left",
                     fontsize=14, fontweight="bold", rotation=90)

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEG_COLORS[i + 1], label=LABELSTONAME[i + 1])
               for i in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10)

    # dice scores — shown in top-right corner when GT is available
    if gt is not None:
        scores     = dice_3d(pred, gt.astype(np.int64), num_classes=5)
        class_names = [LABELSTONAME[i] for i in range(1, 5)]
        lines = ["3D Dice"] + [
            f"{name}: {d:.3f}" if not np.isnan(d) else f"{name}: N/A"
            for name, d in zip(class_names, scores)
        ] + [f"mean: {np.nanmean(scores):.3f}"]
        fig.text(0.99, 0.99, "\n".join(lines), va="top", ha="right",
                 fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--pred",     required=True, help="Path to prediction .nii.gz")
    parser.add_argument("--modality", default="t1c", choices=["t1c", "t1n", "t2f", "t2w"],
                        help="Background MRI modality (default: t1c)")
    parser.add_argument("--gt",       default=None, help="Path to GT seg .nii.gz (optional)")
    parser.add_argument("--save",     default=None, help="Path to save the figure (e.g. result.png)")
    args = parser.parse_args()
    show(args.case_dir, args.pred, args.modality, args.gt, args.save)


if __name__ == "__main__":
    main()
