import numpy as np
import nibabel as nib

shape = (64,64,64)
arr = np.zeros(shape, dtype=np.float32)
img = nib.Nifti1Image(arr, np.eye(4))

filenames = ['flair_test.nii.gz','t1_test.nii.gz','t1ce_test.nii.gz','t2_test.nii.gz']
for fn in filenames:
    nib.save(img, fn)

print('Created files:', filenames)
