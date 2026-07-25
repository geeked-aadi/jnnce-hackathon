import os
import torch
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    EnsureTyped,
    DivisiblePadd,
    CenterSpatialCrop,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Model
# -----------------------------
model = SegResNet(
    spatial_dims=3,
    init_filters=16,
    in_channels=4,
    out_channels=3,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
).to(DEVICE)

weights = torch.load("final_model.pt", map_location=DEVICE)
model.load_state_dict(weights)
model.eval()


# -----------------------------
# Preprocessing
# -----------------------------
transforms = Compose([
    LoadImaged(keys="image"),
    NormalizeIntensityd(
        keys="image",
        nonzero=True,
        channel_wise=True,
    ),
    EnsureTyped(keys="image"),
    DivisiblePadd(keys="image", k=8),
])


import base64
from io import BytesIO


def generate_preview(flair_path, mask):
    """
    Builds two base64 PNGs of the middle axial slice:
    1) the raw FLAIR slice
    2) the same slice with the predicted tumor mask overlaid in color
    """
    flair_img = nib.load(flair_path).get_fdata()

    mid_slice = flair_img.shape[2] // 2

    flair_slice = flair_img[:, :, mid_slice]
    mask_slice = mask[:, :, mid_slice]

    # Normalize flair for display
    flair_slice = (flair_slice - flair_slice.min()) / (
        flair_slice.max() - flair_slice.min() + 1e-8
    )

    overlay = np.zeros((*mask_slice.shape, 3), dtype=np.float32)
    overlay[mask_slice == 1] = [1, 1, 0]   # tumor core - yellow
    overlay[mask_slice == 2] = [0, 1, 0]   # whole tumor (edema) - green
    overlay[mask_slice == 4] = [1, 0, 0]   # enhancing tumor - red

    def fig_to_base64(render_fn):
        fig, ax = plt.subplots(figsize=(5, 5))
        render_fn(ax)
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    original_base64 = fig_to_base64(
        lambda ax: ax.imshow(np.rot90(flair_slice), cmap="gray")
    )

    def render_overlay(ax):
        ax.imshow(np.rot90(flair_slice), cmap="gray")
        ax.imshow(np.rot90(overlay), alpha=0.4)

    overlay_base64 = fig_to_base64(render_overlay)

    return original_base64, overlay_base64


LABELS = {
    1: {"name": "Tumor Core", "color": "#facc15"},       # yellow
    2: {"name": "Whole Tumor (Edema)", "color": "#22c55e"},  # green
    4: {"name": "Enhancing Tumor", "color": "#ef4444"},  # red
}


def compute_label_stats(mask, affine):
    """
    Returns per-class voxel counts and physical volume (cm^3) present in the mask.
    Only includes classes that actually appear in the prediction.
    """
    voxel_dims = np.abs(np.diag(affine)[:3])  # mm per voxel in x, y, z
    voxel_volume_mm3 = float(np.prod(voxel_dims))

    stats = []
    total_voxels = mask.size

    for class_id, info in LABELS.items():
        voxel_count = int(np.sum(mask == class_id))
        if voxel_count == 0:
            continue
        volume_cm3 = (voxel_count * voxel_volume_mm3) / 1000.0
        stats.append({
            "label": class_id,
            "name": info["name"],
            "color": info["color"],
            "voxel_count": voxel_count,
            "volume_cm3": round(volume_cm3, 2),
            "percent_of_volume": round(100 * voxel_count / total_voxels, 3),
        })

    return stats


def predict(flair, t1, t1ce, t2):

    data = {
        "image": [t1ce, t1, t2, flair]
    }

    reference = nib.load(flair)
    original_shape = reference.shape  # (D, H, W) before padding

    data = transforms(data)

    image = data["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)

    pred = torch.sigmoid(output)
    pred = (pred > 0.5).float()

    mask = torch.where(
        pred[:, 2] > 0,
        4,
        torch.where(
            pred[:, 0] > 0,
            1,
            torch.where(pred[:, 1] > 0, 2, 0),
        ),
    )

    mask = mask.squeeze()  # remove batch dim, shape now (D_pad, H_pad, W_pad)
    mask = CenterSpatialCrop(roi_size=original_shape)(mask.unsqueeze(0)).squeeze(0)
    mask = mask.cpu().numpy().astype(np.uint8)

    out = nib.Nifti1Image(
        mask,
        affine=reference.affine,
        header=reference.header
    )

    output_path = "outputs/prediction.nii.gz"

    nib.save(out, output_path)

    preview_base64, overlay_base64 = generate_preview(flair, mask)
    label_stats = compute_label_stats(mask, reference.affine)

    return {
        "nifti_path": output_path,
        "preview_base64": preview_base64,
        "overlay_base64": overlay_base64,
        "label_stats": label_stats,
    }