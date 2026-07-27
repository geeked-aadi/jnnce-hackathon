import numpy as np
import nibabel as nib

try:
    import ants
except Exception:  # pragma: no cover - optional dependency
    ants = None

try:
    from nilearn import datasets
except Exception:  # pragma: no cover - optional dependency
    datasets = None


_atlas_cache = {}


def _get_talairach_atlas():
    """Fetch and cache the Talairach lobe-level atlas when nilearn is available."""
    if "lobe" not in _atlas_cache:
        if datasets is None:
            raise RuntimeError("nilearn is required for anatomical atlas registration")
        atlas = datasets.fetch_atlas_talairach("lobe")
        _atlas_cache["lobe"] = atlas
    return _atlas_cache["lobe"]


def register_and_label(subject_nifti_path):
    """
    Registers the subject scan to MNI space, then warps the Talairach lobe
    atlas back into the subject's native space so it aligns voxel-for-voxel
    with the prediction mask.

    Returns: (atlas_data: np.ndarray, label_names: dict[int, str]) both in
    native subject space, same shape as the subject scan.
    """
    if ants is None or datasets is None:
        subject_img = nib.load(subject_nifti_path)
        return np.zeros(subject_img.shape, dtype=np.int32), {}

    atlas = _get_talairach_atlas()
    atlas_img = nib.load(atlas.maps) if isinstance(atlas.maps, str) else atlas.maps
    label_names = {i: name for i, name in enumerate(atlas.labels) if name and name != "Background"}

    subject_img = ants.image_read(subject_nifti_path)
    template = ants.image_read(str(datasets.load_mni152_template().get_filename())) \
        if hasattr(datasets.load_mni152_template(), "get_filename") else None

    mni_template = datasets.load_mni152_template()
    mni_path = "mni_template_tmp.nii.gz"
    nib.save(mni_template, mni_path)
    template = ants.image_read(mni_path)

    # Register subject -> MNI (affine + deformable)
    reg = ants.registration(fixed=template, moving=subject_img, type_of_transform="SyN")

    # Save atlas to disk for ANTs, then warp MNI-space atlas INTO subject space
    atlas_path = "talairach_atlas_tmp.nii.gz"
    nib.save(atlas_img, atlas_path)
    atlas_ants = ants.image_read(atlas_path)

    warped_atlas = ants.apply_transforms(
        fixed=subject_img,
        moving=atlas_ants,
        transformlist=reg["invtransforms"],
        interpolator="nearestNeighbor",  # preserve discrete label IDs
        whichtoinvert=[True, False],
    )

    atlas_native = warped_atlas.numpy().astype(np.int32)
    return atlas_native, label_names


def tumor_region_overlap(mask, atlas_native, label_names, label_id):
    """
    For a given tumor sub-region (e.g. label_id=4 for enhancing tumor),
    returns the anatomical lobes it overlaps, ranked by voxel overlap.
    """
    tumor_voxels = mask == label_id
    if not np.any(tumor_voxels):
        return []

    overlapping_ids = atlas_native[tumor_voxels]
    unique_ids, counts = np.unique(overlapping_ids, return_counts=True)

    results = []
    total = counts.sum()
    for uid, count in sorted(zip(unique_ids, counts), key=lambda x: -x[1]):
        name = label_names.get(int(uid))
        if not name:
            continue
        results.append({
            "region": name,
            "voxel_count": int(count),
            "percent_of_tumor": round(100 * count / total, 1),
        })
    return results