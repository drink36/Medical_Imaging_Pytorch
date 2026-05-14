"""
Run inference on a single case or a directory of cases.

Usage:
  # single case, no GT
  python infer.py --case_dir path/to/BraTS-GLI-00005-100

  # single case with external GT seg for Dice
  python infer.py --case_dir path/to/BraTS-GLI-00005-100 --gt path/to/BraTS-GLI-00005-100-seg.nii.gz

  # all cases in a directory, GT segs in a separate folder
  python infer.py --case_dir path/to/validation_data --batch_cases --gt_dir path/to/gt_segs
"""

import os
import argparse
import glob
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader, TensorDataset

from unet import UNet
from load import MODALITIES, load_nifti, normalize_nonzero, get_case_paths, LABELSTONAME


# ── Checkpoint ────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    cfg   = ckpt["cfg"]
    model = UNet(in_channels=cfg["in_channels"], out_channels=cfg["num_classes"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_dice={ckpt['val_dice']:.4f}")
    return model, cfg


# ── Predict one case ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_case(model, case_dir: str, cfg: dict, device: torch.device, gt_path: str = None) -> tuple:
    """Returns (case_id, pred_vol [H,W,D], gt_vol [H,W,D] or None, ref_nii_path)."""
    case_id, paths = get_case_paths(case_dir)

    volumes = {m: load_nifti(paths[m])[0] for m in MODALITIES}

    # resolve GT: explicit path > same case dir > None
    _gt_path = gt_path or (paths["seg"] if os.path.exists(paths["seg"]) else None)
    gt_vol   = load_nifti(_gt_path)[0].astype(np.int64) if _gt_path else None

    # Stack modalities → [D, 4, H, W]  (same layout as training)
    img = np.stack(
        [normalize_nonzero(volumes[m]) for m in MODALITIES], axis=0
    ).transpose(3, 0, 1, 2).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(img)),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    slices = []
    for (batch,) in loader:
        preds = model(batch.to(device)).argmax(dim=1).cpu().numpy()  # [B, H, W]
        slices.append(preds)

    pred_vol = np.concatenate(slices, axis=0).transpose(1, 2, 0)  # [H, W, D]
    return case_id, pred_vol, gt_vol, paths["t1c"]


# ── Dice (3D, per class) ──────────────────────────────────────────────────────
def dice_3d(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> list:
    scores = []
    for c in range(1, num_classes):
        p = (pred == c).astype(np.float32)
        t = (gt   == c).astype(np.float32)
        denom = p.sum() + t.sum()
        scores.append(float("nan") if denom == 0 else float(2 * (p * t).sum() / denom))
    return scores


# ── Save as NIfTI ─────────────────────────────────────────────────────────────
def save_nifti(pred_vol: np.ndarray, ref_path: str, out_path: str) -> None:
    ref = nib.load(ref_path)
    nib.save(nib.Nifti1Image(pred_vol.astype(np.uint8), ref.affine, ref.header), out_path)
    print(f"  Saved → {out_path}")


# ── Run ───────────────────────────────────────────────────────────────────────
def run(case_dir: str, model, cfg: dict, device: torch.device, out_dir: str, gt_path: str = None) -> None:
    print(f"\nCase: {os.path.basename(case_dir)}")
    case_id, pred_vol, gt_vol, ref_path = predict_case(model, case_dir, cfg, device, gt_path)

    os.makedirs(out_dir, exist_ok=True)
    save_nifti(pred_vol, ref_path, os.path.join(out_dir, f"{case_id}-pred.nii.gz"))

    if gt_vol is not None:
        scores = dice_3d(pred_vol, gt_vol, cfg["num_classes"])
        class_names = [LABELSTONAME[i] for i in range(1, cfg["num_classes"])]
        print("  3D Dice:")
        for name, d in zip(class_names, scores):
            print(f"    {name:<8}: {d:.4f}" if not np.isnan(d) else f"    {name:<8}: N/A (absent)")
        print(f"    mean    : {np.nanmean(scores):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir",    required=True, help="Case dir or root dir (with --batch_cases)")
    parser.add_argument("--ckpt",        default="checkpoints/best.pth")
    parser.add_argument("--out_dir",     default="predictions")
    parser.add_argument("--batch_cases", action="store_true", help="Run all subdirs in case_dir")
    parser.add_argument("--gt",          default=None, help="Path to a single GT seg .nii.gz")
    parser.add_argument("--gt_dir",      default=None, help="Dir of GT segs named {case_id}-seg.nii.gz (for --batch_cases)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_checkpoint(args.ckpt, device)

    if args.batch_cases:
        case_dirs = sorted(d for d in glob.glob(os.path.join(args.case_dir, "*")) if os.path.isdir(d))
        print(f"Found {len(case_dirs)} cases")
        for case_dir in case_dirs:
            try:
                case_id  = os.path.basename(case_dir)
                gt_path  = os.path.join(args.gt_dir, f"{case_id}-seg.nii.gz") if args.gt_dir else None
                run(case_dir, model, cfg, device, args.out_dir, gt_path)
            except Exception as e:
                print(f"  SKIP {case_dir}: {e}")
    else:
        run(args.case_dir, model, cfg, device, args.out_dir, args.gt)


if __name__ == "__main__":
    main()
