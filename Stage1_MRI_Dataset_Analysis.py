"""
Stage 1 - MRI Dataset Exploration, Analysis & Preparation
MedhaDrishti AI Hackathon - Medical Image Enhancement & Segmentation

Handles the REAL nested folder structure discovered on disk:

Brain_JN_1/Brain DATASETS/
    Normal brain Datasets/S1/2D MRI/*.nii...    Normal brain Datasets/S1/3D MRI/*.nii...
    Pathological brain MRI Datasets/BRP1/*.nii...

Spine DATASETS/Spine DATASETS/
    Normal Spine MRI Datasets/SP1/2D T1 MRI/*.nii...   .../2D T2 MRI/*.nii...
    Pathological Spine MRI Datasets/SP11/*.nii...

Instead of assuming one folder = one patient (like BRATS), this version
RECURSIVELY WALKS every file under the root, and infers:
    - condition   (Normal / Pathological)  from the path
    - patient_id  (S1, BRP1, SP1, SP11...) from the path
    - modality    (T1 / T2 / FLAIR / STIR / T1c / unknown) from filename+folder text
    - dimensionality (2D / 3D)              from folder name

Run locally with:
    pip install nibabel opencv-python scikit-image scikit-learn pandas matplotlib seaborn
    python Stage1_MRI_Dataset_Analysis.py

Before running, edit BRAIN_ROOT and SPINE_ROOT below.

Outputs (saved to ./stage1_outputs/):
    - brain_mri_image_properties.csv
    - spine_mri_image_properties.csv
    - dataset_split.csv
    - brain_modality_comparison.png
    - spine_modality_comparison.png
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import laplace
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 4)

# =============================================================================
# 1. CONFIG - EDIT THESE PATHS to match your local folders
# =============================================================================
BRAIN_ROOT = r"./Brain_JN_1/Brain DATASETS"
SPINE_ROOT = r"./Spine DATASETS/Spine DATASETS"

OUTPUT_DIR = "./stage1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NIFTI_EXTENSIONS = ('.nii', '.nii.gz')

print('Config set. Output will be saved to:', OUTPUT_DIR)


# =============================================================================
# 2. RECURSIVE DISCOVERY - find every .nii/.nii.gz file, wherever it is
# =============================================================================
def find_all_nifti_files(root_dir):
    """Recursively find every NIfTI file under root_dir, at any depth."""
    if not os.path.isdir(root_dir):
        print(f'[!] Path not found: {root_dir}')
        return []
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(NIFTI_EXTENSIONS):
                matches.append(os.path.join(dirpath, fname))
    return sorted(matches)


def parse_condition(rel_path):
    """Infer Normal vs Pathological from any folder name in the path."""
    low = rel_path.lower()
    if 'patholog' in low:
        return 'Pathological'
    elif 'normal' in low:
        return 'Normal'
    return 'Unknown'


def parse_patient_id(rel_path):
    """The patient folder is whatever comes right after the Normal/Pathological folder.
    e.g. 'Normal brain Datasets/S1/2D MRI/file.nii' -> 'S1'
         'Pathological Spine MRI Datasets/SP11/file.nii' -> 'SP11'
    Falls back to the first path component if no Normal/Pathological folder is found."""
    parts = rel_path.split(os.sep)
    for i, p in enumerate(parts):
        if 'normal' in p.lower() or 'patholog' in p.lower():
            return parts[i + 1] if i + 1 < len(parts) else parts[i]
    return parts[0] if parts else 'unknown'


def parse_dimensionality(rel_path):
    low = rel_path.lower()
    if '2d' in low:
        return '2D'
    elif '3d' in low:
        return '3D'
    return 'unknown'


def parse_modality(rel_path):
    """Detect modality from filename + folder names combined.
    Checked in order of specificity so 'flair' isn't mistaken for 't1', etc."""
    low = rel_path.lower().replace('_', ' ').replace('-', ' ')
    if 'flair' in low:
        return 'FLAIR'
    if 'stir' in low:
        return 'STIR'
    if 't1c' in low or 't1ce' in low or 'contrast' in low or 't1 c' in low:
        return 'T1c'
    if 't2' in low:
        return 'T2'
    if 't1' in low:
        return 'T1'
    return 'unknown'


print('Discovery utilities ready')


# =============================================================================
# 3. VOLUME LOADING UTILITIES
# =============================================================================
def load_nifti(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data, img


def get_representative_slice(volume):
    """Return a single 2D slice regardless of whether the file is already
    a 2D image, a 3D volume, or has a stray singleton 4th dimension."""
    vol = np.squeeze(volume)
    if vol.ndim == 2:
        return vol
    elif vol.ndim == 3:
        idx = vol.shape[2] // 2
        return vol[:, :, idx]
    elif vol.ndim >= 4:
        vol3 = vol.reshape(vol.shape[0], vol.shape[1], -1)
        idx = vol3.shape[2] // 2
        return vol3[:, :, idx]
    return vol


def normalize_uint8(slice_2d):
    s = slice_2d.astype(np.float32)
    s -= s.min()
    if s.max() > 0:
        s = s / s.max() * 255.0
    return s.astype(np.uint8)


print('Loading utilities ready')


# =============================================================================
# 4. IMAGE PROPERTY METRICS (Contrast, Complexity, Sharpness, Edge, Noise, Mean, Deviation)
# =============================================================================
def estimate_noise(slice_2d):
    """Fast noise estimate via Laplacian-based method (Immerkaer's formula)."""
    H, W = slice_2d.shape
    if H < 3 or W < 3:
        return 0.0
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    conv = cv2.filter2D(slice_2d.astype(np.float64), -1, M)
    sigma = np.sum(np.abs(conv))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return sigma


def compute_image_properties(slice_2d):
    img = slice_2d.astype(np.float32)

    mean_val = np.mean(img)
    std_val = np.std(img)                                   # 'Deviation'
    contrast = float(img.max() - img.min()) if img.size else 0.0
    rms_contrast = std_val / (mean_val + 1e-8)
    sharpness = laplace(img).var()
    edge_strength = float(np.mean(cv2.Canny(slice_2d.astype(np.uint8), 50, 150) > 0))
    complexity = shannon_entropy(slice_2d.astype(np.uint8))
    noise = estimate_noise(slice_2d)

    return {
        'mean': mean_val,
        'std_dev': std_val,
        'contrast_range': contrast,
        'rms_contrast': rms_contrast,
        'sharpness_laplacian_var': sharpness,
        'edge_strength': edge_strength,
        'complexity_entropy': complexity,
        'noise_level': noise
    }


print('Metric functions ready')


# =============================================================================
# 5. GENERIC DATASET ANALYSIS (works for both Brain and Spine given a root)
# =============================================================================
def analyze_dataset(root_dir, label):
    files = find_all_nifti_files(root_dir)
    print(f'{label}: found {len(files)} NIfTI files under {root_dir}')

    records = []
    for filepath in files:
        rel_path = os.path.relpath(filepath, root_dir)
        try:
            volume, img_obj = load_nifti(filepath)
            voxel_spacing = img_obj.header.get_zooms()
            slice_2d = get_representative_slice(volume)
            if slice_2d.ndim != 2 or slice_2d.size == 0:
                print(f'[!] Skipping (unexpected shape {volume.shape}): {rel_path}')
                continue
            slice_u8 = normalize_uint8(slice_2d)
            props = compute_image_properties(slice_u8)

            record = {
                'patient_id': parse_patient_id(rel_path),
                'condition': parse_condition(rel_path),
                'dimensionality': parse_dimensionality(rel_path),
                'modality': parse_modality(rel_path),
                'shape': volume.shape,
                'voxel_spacing': voxel_spacing,
                'filepath': filepath
            }
            record.update(props)
            records.append(record)
        except Exception as e:
            print(f'[!] Failed on {rel_path}: {e}')

    df = pd.DataFrame(records)
    print(f'{label}: successfully analyzed {len(df)} files')
    return df


brain_df = analyze_dataset(BRAIN_ROOT, 'BRAIN')
spine_df = analyze_dataset(SPINE_ROOT, 'SPINE')

print()
if not brain_df.empty:
    print('--- Brain modality/condition breakdown ---')
    print(brain_df.groupby(['condition', 'modality']).size())
if not spine_df.empty:
    print('--- Spine modality/condition breakdown ---')
    print(spine_df.groupby(['condition', 'modality']).size())


# =============================================================================
# 6. SUMMARY STATISTICS TABLE (per modality) - key deliverable
# =============================================================================
metric_cols = ['mean', 'std_dev', 'contrast_range', 'rms_contrast',
               'sharpness_laplacian_var', 'edge_strength', 'complexity_entropy', 'noise_level']


def summarize_by_modality(df, label):
    if df.empty:
        print(f'{label}: no data loaded - check dataset path')
        return pd.DataFrame()
    summary = df.groupby('modality')[metric_cols].agg(['mean', 'std']).round(3)
    print(f'--- {label}: summary by modality ---')
    print(summary)
    return summary


def summarize_by_condition(df, label):
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby(['condition', 'modality'])[metric_cols].agg(['mean', 'std']).round(3)
    print(f'--- {label}: summary by condition + modality ---')
    print(summary)
    return summary


brain_summary = summarize_by_modality(brain_df, 'BRAIN MRI')
spine_summary = summarize_by_modality(spine_df, 'SPINE MRI')

brain_cond_summary = summarize_by_condition(brain_df, 'BRAIN MRI')
spine_cond_summary = summarize_by_condition(spine_df, 'SPINE MRI')


# =============================================================================
# 7. VISUAL COMPARISON ACROSS SUB-MODALITIES
# =============================================================================
def plot_metric_comparison(df, label):
    if df.empty:
        return
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, metric in enumerate(metric_cols):
        sns.boxplot(data=df, x='modality', y=metric, ax=axes[i])
        axes[i].set_title(metric)
        axes[i].tick_params(axis='x', rotation=45)
    fig.suptitle(f'{label}: Image property comparison across modalities', fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f'{label.lower()}_modality_comparison.png')
    plt.savefig(outpath, dpi=150)
    print(f'Saved plot: {outpath}')
    plt.close(fig)


plot_metric_comparison(brain_df, 'Brain')
plot_metric_comparison(spine_df, 'Spine')


# =============================================================================
# 8. SAMPLE SLICE VISUALIZATION (qualitative spot-check, one file per modality)
# =============================================================================
def show_sample_slices(df, label, n=6):
    if df.empty:
        return
    sample = df.groupby('modality').first().reset_index().head(n)
    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, sample.iterrows()):
        volume, _ = load_nifti(row['filepath'])
        slice_2d = get_representative_slice(volume)
        ax.imshow(slice_2d.T, cmap='gray', origin='lower')
        ax.set_title(f"{row['modality']}\n{row['condition']} - {row['patient_id']}")
        ax.axis('off')
    plt.suptitle(f'{label}: sample slice per modality')
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f'{label.lower()}_sample_slices.png')
    plt.savefig(outpath, dpi=150)
    print(f'Saved plot: {outpath}')
    plt.close(fig)


show_sample_slices(brain_df, 'Brain')
show_sample_slices(spine_df, 'Spine')


# =============================================================================
# 9. TRAIN / TEST / VALIDATION SPLIT (patient-level, stratified by condition)
# =============================================================================
def make_patient_split(df, train_size=0.7, val_size=0.15, seed=42):
    if df.empty:
        return pd.DataFrame(columns=['patient_id', 'condition', 'split'])

    patients = df[['patient_id', 'condition']].drop_duplicates().reset_index(drop=True)
    n = len(patients)

    # Too few patients for a meaningful 3-way split -> everything goes to train
    if n < 6:
        patients['split'] = 'train'
        print(f'Only {n} unique patients found - using all as train for now')
        return patients

    def safe_split(data, size, seed, stratify_col=None):
        """train_test_split wrapper that falls back to unstratified split
        when a class is too small to stratify."""
        stratify = data[stratify_col] if stratify_col else None
        try:
            return train_test_split(data, train_size=size, random_state=seed, stratify=stratify)
        except ValueError:
            return train_test_split(data, train_size=size, random_state=seed)

    train, temp = safe_split(patients, train_size, seed, 'condition')
    rel_val = val_size / (1 - train_size)

    if len(temp) < 2:
        # not enough left to split into val/test -> put remainder into val
        val, test = temp, temp.iloc[0:0]
    else:
        val, test = safe_split(temp, rel_val, seed, 'condition')

    train = train.copy(); train['split'] = 'train'
    val = val.copy(); val['split'] = 'val'
    test = test.copy(); test['split'] = 'test'
    return pd.concat([train, val, test], ignore_index=True)


brain_split = make_patient_split(brain_df)
brain_split['organ'] = 'brain'
spine_split = make_patient_split(spine_df)
spine_split['organ'] = 'spine'

split_df = pd.concat([brain_split, spine_split], ignore_index=True)
print()
print('--- Dataset split summary ---')
if not split_df.empty:
    print(split_df.groupby(['organ', 'split']).size())

split_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_split.csv'), index=False)


# =============================================================================
# 10. EXPORT FINAL STAGE 1 DELIVERABLES
# =============================================================================
if not brain_df.empty:
    brain_df.to_csv(os.path.join(OUTPUT_DIR, 'brain_mri_image_properties.csv'), index=False)
if not spine_df.empty:
    spine_df.to_csv(os.path.join(OUTPUT_DIR, 'spine_mri_image_properties.csv'), index=False)

print()
print('Saved deliverables to:', OUTPUT_DIR)
print(sorted(os.listdir(OUTPUT_DIR)))
