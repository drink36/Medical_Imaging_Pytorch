import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MODALITIES = ["t1c", "t1n", "t2f", "t2w"]


LABELSTONAME = {
    0: "background",
    1: "NETC", 
    2: "SNFH", 
    3: "ET", 
    4: "RC" 
}
def load_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    spacing = header.get_zooms()[:3]
    return data, affine, header, spacing

def get_case_paths(case_dir):
    case_id = os.path.basename(os.path.normpath(case_dir))
    paths = {m: os.path.join(case_dir, f"{case_id}-{m}.nii.gz") for m in MODALITIES}
    paths["seg"] = os.path.join(case_dir, f"{case_id}-seg.nii.gz")
    return case_id, paths

def load_case(case_dir):
    case_id, paths = get_case_paths(case_dir)
    volumes = {}
    spacings = {}

    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
        data, affine, header, spacing = load_nifti(p)
        volumes[k] = data
        spacings[k] = spacing
    return case_id, volumes, spacings

def check_case(volumes, spacings):
    print("=== Shape Check ===")
    for k, v in volumes.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")

    print("\n=== Spacing Check ===")
    for k, s in spacings.items():
        print(f"{k}: spacing={s}")

    shapes = {k: v.shape for k, v in volumes.items()}
    same_shape = len(set(shapes.values())) == 1
    print("\nAll shapes same:", same_shape)

    print("\nSeg labels:", np.unique(volumes["seg"]))

def get_tumor_center(seg):
    coords = np.argwhere(seg > 0)
    if len(coords) == 0:
        # fallback: volume center
        return tuple(np.array(seg.shape) // 2)
    center = coords.mean(axis=0).astype(int)
    return tuple(center)

def get_best_slices(seg):
    mask = seg > 0
    if mask.sum() == 0:
        return tuple(np.array(seg.shape) // 2)

    x = int(np.argmax(mask.sum(axis=(1, 2))))
    y = int(np.argmax(mask.sum(axis=(0, 2))))
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    return x, y, z

def normalize_nonzero(x):
    x = x.copy()
    mask = x != 0
    if mask.sum() > 0:
        mean = x[mask].mean()
        std = x[mask].std()
        if std > 0:
            x[mask] = (x[mask] - mean) / std
    return x

def select_one_label(seg, target_label):
    if isinstance(target_label, (list, tuple, set, np.ndarray)):
        return np.isin(seg, list(target_label)).astype(np.uint8)
    return (seg == target_label).astype(np.uint8)

def draw_bbox_2d(ax, seg2d, color="lime", lw=1.8):
    mask = seg2d.T > 0
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    rect = Rectangle(
        (cmin, rmin),
        cmax - cmin + 1,
        rmax - rmin + 1,
        fill=False,
        edgecolor=color,
        linewidth=lw,
    )
    ax.add_patch(rect)

def plot_three_views(volumes, modality="t1c", alpha=0.35, normalize=True):
    img = volumes[modality]
    seg = volumes["seg"]

    if normalize:
        img = normalize_nonzero(img)

    x, y, z = get_tumor_center(seg)
    print(f"Tumor center (x, y, z) = {(x, y, z)}")

    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    fig.suptitle(f"{modality} with seg overlay", fontsize=14)

    # Axial: z fixed
    axial_img = img[:, :, z]
    axial_seg = seg[:, :, z]
    axes[0].imshow(axial_img.T, cmap="gray", origin="lower")
    axes[0].imshow(np.ma.masked_where(axial_seg.T == 0, axial_seg.T),
                   cmap="jet", alpha=alpha, origin="lower")
    axes[0].set_title(f"Axial (z={z})")
    axes[0].axis("off")

    # Coronal: y fixed
    coronal_img = img[:, y, :]
    coronal_seg = seg[:, y, :]
    axes[2].imshow(coronal_img.T, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(coronal_seg.T == 0, coronal_seg.T),
                   cmap="jet", alpha=alpha, origin="lower")
    axes[2].set_title(f"Coronal (y={y})")
    axes[2].axis("off")

    # Sagittal: x fixed
    sagittal_img = img[x, :, :]
    sagittal_seg = seg[x, :, :]
    axes[4].imshow(sagittal_img.T, cmap="gray", origin="lower")
    axes[4].imshow(np.ma.masked_where(sagittal_seg.T == 0, sagittal_seg.T),
                   cmap="jet", alpha=alpha, origin="lower")
    axes[4].set_title(f"Sagittal (x={x})")
    axes[4].axis("off")

    # without seg overlay
    axes[1].imshow(axial_img.T, cmap="gray", origin="lower")
    axes[1].set_title(f"Axial (z={z})")
    axes[1].axis("off")
    axes[3].imshow(coronal_img.T, cmap="gray", origin="lower")
    axes[3].set_title(f"Coronal (y={y})")
    axes[3].axis("off")
    axes[5].imshow(sagittal_img.T, cmap="gray", origin="lower")
    axes[5].set_title(f"Sagittal (x={x})")
    axes[5].axis("off")
    plt.tight_layout()
    plt.show()

def plot_modalities_three_views_single_label(
    volumes,
    target_label=1,
    alpha=0.35,
    normalize=True,
    slice_mode="max_area",
):
    seg = select_one_label(volumes["seg"], target_label)
    selected_voxels = int(seg.sum())
    print(f"Single label={target_label} | selected voxels={selected_voxels}")
    if selected_voxels == 0:
        print("No voxels matched target_label; overlay will be empty.")

    if slice_mode == "center":
        x, y, z = get_tumor_center(seg)
    else:
        x, y, z = get_best_slices(seg)

    fig, axes = plt.subplots(len(MODALITIES), 6, figsize=(15, 18))
    fig.suptitle(f"Single label {target_label} across modalities", fontsize=16)

    for row, mod in enumerate(MODALITIES):
        img = volumes[mod]
        if normalize:
            img = normalize_nonzero(img)

        views = [
            (img[:, :, z], seg[:, :, z], f"{mod} Axial z={z}"),
            (img[:, y, :], seg[:, y, :], f"{mod} Coronal y={y}"),
            (img[x, :, :], seg[x, :, :], f"{mod} Sagittal x={x}")
        ]

        for col, (img2d, seg2d, title) in enumerate(views):
            seg_overlay = np.ma.masked_where(seg2d.T == 0, np.ones_like(seg2d.T))
            axes[row, col * 2].imshow(img2d.T, cmap="gray", origin="lower")
            axes[row, col * 2].imshow(
                seg_overlay,
                cmap="autumn",
                alpha=max(alpha, 0.7),
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[row, col * 2].contour((seg2d.T > 0).astype(np.uint8), levels=[0.5], colors="cyan", linewidths=1.2)
            draw_bbox_2d(axes[row, col * 2], seg2d)
            axes[row, col * 2].set_title(title)
            axes[row, col * 2].axis("off")

            axes[row, col * 2 + 1].imshow(img2d.T, cmap="gray", origin="lower")
            axes[row, col * 2 + 1].set_title(title + " (image only)")
            draw_bbox_2d(axes[row, col * 2 + 1], seg2d)
            axes[row, col * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_modalities_three_views_multi_label(
    volumes,
    alpha=0.35,
    normalize=True,
    slice_mode="max_area",
):
    seg = volumes["seg"]

    if slice_mode == "center":
        x, y, z = get_tumor_center(seg)
    else:
        x, y, z = get_best_slices(seg)

    fig, axes = plt.subplots(len(MODALITIES), 3, figsize=(15, 18))
    fig.suptitle("Multi-label overlay across modalities", fontsize=16)

    for row, mod in enumerate(MODALITIES):
        img = volumes[mod]
        if normalize:
            img = normalize_nonzero(img)

        views = [
            (img[:, :, z], seg[:, :, z], f"{mod} Axial z={z}"),
            (img[:, y, :], seg[:, y, :], f"{mod} Coronal y={y}"),
            (img[x, :, :], seg[x, :, :], f"{mod} Sagittal x={x}")
        ]

        for col, (img2d, seg2d, title) in enumerate(views):
            axes[row, col].imshow(img2d.T, cmap="gray", origin="lower")
            axes[row, col].imshow(np.ma.masked_where(seg2d.T == 0, seg2d.T),
                                  cmap="jet", alpha=alpha, origin="lower")
            axes[row, col].set_title(title)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def summarize_intensity(volumes):
    print("=== Intensity Summary (non-zero voxels only) ===")
    for mod in MODALITIES:
        x = volumes[mod]
        nz = x[x != 0]
        if len(nz) == 0:
            print(f"{mod}: all zeros")
            continue

        print(
            f"{mod}: "
            f"count={len(nz)}, "
            f"min={nz.min():.3f}, "
            f"max={nz.max():.3f}, "
            f"mean={nz.mean():.3f}, "
            f"std={nz.std():.3f}, "
            f"p1={np.percentile(nz, 1):.3f}, "
            f"p99={np.percentile(nz, 99):.3f}"
        )

def plot_intensity_histograms(volumes, bins=100):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, mod in enumerate(MODALITIES):
        x = volumes[mod]
        nz = x[x != 0]

        axes[i].hist(nz, bins=bins)
        axes[i].set_title(mod)
        axes[i].set_xlabel("Intensity")
        axes[i].set_ylabel("Voxel count")

    plt.tight_layout()
    plt.show()

def summarize_segmentation(seg):
    labels, counts = np.unique(seg, return_counts=True)
    print("Unique labels in seg:", labels)
    print("=== Segmentation Label Summary ===")
    total_voxels = seg.size
    tumor_voxels = np.sum(seg > 0)

    for label, count in zip(labels, counts):
        label_name = LABELSTONAME.get(int(label), f"Label {int(label)}")
        print(f"{label_name}: count={int(count)} ratio={count/total_voxels:.6f}")

    print(f"\nTotal voxels: {total_voxels}")
    print(f"Tumor voxels (>0): {int(tumor_voxels)}")
    print(f"Tumor ratio: {tumor_voxels / total_voxels:.6f}")

def tumor_bounding_box(seg):
    coords = np.argwhere(seg > 0)
    if len(coords) == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    size = maxs - mins + 1
    return mins, maxs, size

def summarize_tumor_bbox(seg, spacing=None):
    bbox = tumor_bounding_box(seg)
    if bbox is None:
        print("No tumor found.")
        return

    mins, maxs, size = bbox
    print("=== Tumor Bounding Box ===")
    print(f"min index: {mins}")
    print(f"max index: {maxs}")
    print(f"bbox size (voxels): {size}")

    if spacing is not None:
        physical = size * np.array(spacing)
        print(f"bbox size (mm): {physical}")

def collect_dataset_stats(root_dir):
    case_dirs = sorted(
        [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    )

    rows = []

    for case_dir in case_dirs:
        try:
            case_id, volumes, spacings = load_case(case_dir)
            seg = volumes["seg"]

            row = {"case_id": case_id}
            for mod in MODALITIES:
                row[f"{mod}_shape"] = volumes[mod].shape
                row[f"{mod}_spacing"] = spacings[mod]

                nz = volumes[mod][volumes[mod] != 0]
                if len(nz) > 0:
                    row[f"{mod}_mean"] = float(nz.mean())
                    row[f"{mod}_std"] = float(nz.std())
                else:
                    row[f"{mod}_mean"] = np.nan
                    row[f"{mod}_std"] = np.nan

            labels, counts = np.unique(seg, return_counts=True)
            label_count_map = {int(l): int(c) for l, c in zip(labels, counts)}
            row["seg_labels"] = label_count_map
            row["tumor_voxels"] = int(np.sum(seg > 0))
            row["tumor_ratio"] = float(np.sum(seg > 0) / seg.size)

            bbox = tumor_bounding_box(seg)
            if bbox is not None:
                mins, maxs, size = bbox
                row["bbox_size_voxels"] = tuple(size.tolist())
            else:
                row["bbox_size_voxels"] = None

            rows.append(row)

        except Exception as e:
            print(f"Skip {case_dir}: {e}")

    return rows



# main
if __name__ == "__main__":
    case_dir = "BraTS2024-BraTS-GLI-TrainingData\\training_data1_v2\\BraTS-GLI-02112-100"
    

    case_id, volumes, spacings = load_case(case_dir)
    print("Case:", case_id)

    check_case(volumes, spacings)

    print()
    summarize_intensity(volumes)

    print()
    summarize_segmentation(volumes["seg"])
    # if there's no 1, 2, 3, 4 label in seg, the bbox will be empty, we exit
    for label in [1, 2, 3, 4]:
        if np.sum(volumes["seg"] == label) == 0:
            print(f"No label {label} found in seg, skipping bbox summary.")
        else:
            print(f"\nLabel {label} ({LABELSTONAME.get(label, f'Label {label}')}) bbox summary:")
    summarize_tumor_bbox(volumes["seg"], spacing=spacings["seg"])
    # plot_three_views(volumes, modality="t1n")
    # plot_three_views(volumes, modality="t1c")

    # plot_three_views(volumes, modality="t2f")
    # plot_three_views(volumes, modality="t2w")
    plot_modalities_three_views_single_label(volumes, target_label=1)
    plot_modalities_three_views_single_label(volumes, target_label=2)
    plot_modalities_three_views_single_label(volumes, target_label=3)
    plot_modalities_three_views_single_label(volumes, target_label=4)
    # plot_modalities_three_views_multi_label(volumes, alpha=0.5, normalize=True, slice_mode="max_area")



    # plot_modalities_three_views(volumes)
    # plot_intensity_histograms(volumes)