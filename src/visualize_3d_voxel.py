import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load Ground Truth Mask
# ----------------------------
seg_path = "data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_016/BraTS20_Training_016_seg.nii"

seg = nib.load(seg_path).get_fdata()

# Convert to binary
tumor = (seg > 0)

# Reduce resolution (VERY IMPORTANT for speed)
tumor_small = tumor[::6, ::6, ::6]

# ----------------------------
# Plot voxels
# ----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.voxels(tumor_small, edgecolor='k')

ax.set_title("3D Tumor (Voxel View)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()