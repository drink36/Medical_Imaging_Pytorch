import os
import glob
import random
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import BraTSDataset
from unet import UNet
from load import LABELSTONAME

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "npy_dir":        "C:\\Users\\ooo91\\Desktop\\School\\Medical\\Dataset\\BraTS2024-preprocessed",
    "data_ratio":     1.0,    # fraction
    "val_ratio":      0.2,
    "axis":           2,        # axial slices
    "tumor_only":     True,
    "in_channels":    4,        # t1c, t1n, t2f, t2w
    "num_classes":    5,        # background + NETC + SNFH + ET + RC
    "batch_size":     16,
    "num_workers":    0,        
    "lr":             1e-4,
    "epochs":         20,
    "checkpoint_dir": "checkpoints",
    "seed":           42,
}

# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, background class excluded."""

    def __init__(self, num_classes: int, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : [B, C, H, W]   targets : [B, H, W] long
        probs      = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dice = 0.0
        for c in range(1, self.num_classes):
            p = probs[:, c]
            t = targets_oh[:, c]
            intersection = (p * t).sum()
            dice += (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)

        return 1.0 - dice / (self.num_classes - 1)


# ── Metric ────────────────────────────────────────────────────────────────────
def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list:
    """Hard Dice per foreground class. Returns nan when class absent in both."""
    scores = []
    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        denom  = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            scores.append(float("nan"))
        else:
            scores.append((2.0 * (pred_c * tgt_c).sum() / denom).item())
    return scores


# ── Data ──────────────────────────────────────────────────────────────────────
def split_cases(npy_dir: str, data_ratio: float, val_ratio: float, seed: int):
    seg_files = sorted(glob.glob(os.path.join(npy_dir, "*_seg.npy")))
    case_ids  = [os.path.basename(f)[:-8] for f in seg_files]  # strip _seg.npy
    random.seed(seed)
    random.shuffle(case_ids)
    case_ids = case_ids[:max(2, int(len(case_ids) * data_ratio))]
    n_val = max(1, int(len(case_ids) * val_ratio))
    return case_ids[n_val:], case_ids[:n_val]


def get_loaders(cfg: dict):
    train_ids, val_ids = split_cases(cfg["npy_dir"], cfg["data_ratio"], cfg["val_ratio"], cfg["seed"])
    print(f"Cases  — train: {len(train_ids)}, val: {len(val_ids)}")

    train_ds = BraTSDataset(cfg["npy_dir"], case_ids=train_ids, axis=cfg["axis"], tumor_only=cfg["tumor_only"])
    val_ds   = BraTSDataset(cfg["npy_dir"], case_ids=val_ids,   axis=cfg["axis"], tumor_only=cfg["tumor_only"])
    print(f"Slices — train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


# ── Train / Val loops ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, ce_fn, dice_fn, device) -> float:
    model.train()
    total_loss = 0.0
    bar = tqdm.tqdm(loader, desc="  Train", leave=False, unit="batch")
    for images, masks in bar:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = ce_fn(logits, masks) + dice_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, ce_fn, dice_fn, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_scores = []

    bar = tqdm.tqdm(loader, desc="  Val  ", leave=False, unit="batch")
    for images, masks in bar:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = (ce_fn(logits, masks) + dice_fn(logits, masks)).item()
        total_loss += loss

        preds = logits.argmax(dim=1)
        all_scores.append(dice_score(preds, masks, num_classes))
        bar.set_postfix(loss=f"{loss:.4f}")

    mean_dice = np.nanmean(np.array(all_scores), axis=0)  # [num_classes - 1]
    return total_loss / len(loader), mean_dice


# ── Main ──────────────────────────────────────────────────────────────────────
def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.tqdm.write(f"Device: {device}\n")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])

    train_loader, val_loader = get_loaders(cfg)

    model     = UNet(in_channels=cfg["in_channels"], out_channels=cfg["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    ce_fn   = nn.CrossEntropyLoss()
    dice_fn = DiceLoss(cfg["num_classes"])

    class_names = [LABELSTONAME[i] for i in range(1, cfg["num_classes"])]
    best_dice = -1.0

    epoch_bar = tqdm.tqdm(range(1, cfg["epochs"] + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        train_loss          = train_one_epoch(model, train_loader, optimizer, ce_fn, dice_fn, device)
        val_loss, mean_dice = validate(model, val_loader, ce_fn, dice_fn, device, cfg["num_classes"])
        mean_val_dice       = float(np.nanmean(mean_dice))
        scheduler.step(mean_val_dice)

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", dice=f"{mean_val_dice:.4f}")

        lines = [f"\nEpoch {epoch:3d}/{cfg['epochs']}  train={train_loss:.4f}  val={val_loss:.4f}  dice={mean_val_dice:.4f}"]
        for name, d in zip(class_names, mean_dice):
            lines.append(f"  {name:<8}: {d:.4f}" if not np.isnan(d) else f"  {name:<8}: N/A (absent)")

        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            ckpt_path = os.path.join(cfg["checkpoint_dir"], "best.pth")
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "val_dice": best_dice,
                "cfg":      cfg,
            }, ckpt_path)
            lines.append(f"  Saved checkpoint — best dice={best_dice:.4f}")

        tqdm.tqdm.write("\n".join(lines))


if __name__ == "__main__":
    main(CFG)
