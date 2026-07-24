import nibabel as nib
import numpy as np
import os

def load_mri(file_path):
    image = nib.load(file_path)
    volume = image.get_fdata()
    return volume