import nibabel as nib
import matplotlib.pyplot as plt

# Load FLAIR image
img = nib.load(
    "../data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_016/"
    "BraTS20_Training_016_flair.nii"
).get_fdata()

# Load segmentation mask
mask = nib.load(
    "../data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_016/"
    "BraTS20_Training_016_seg.nii"
).get_fdata()

print("Shape of MRI:", img.shape)

# ✅ DEFINE middle slice index
z_mid = img.shape[2] // 2

# Plot MRI + Mask overlay
plt.figure(figsize=(6, 6))
plt.imshow(img[:, :, z_mid], cmap="gray")
plt.imshow(mask[:, :, z_mid], cmap="jet", alpha=0.4)
plt.title("Tumor Overlay on FLAIR MRI")
plt.axis("off")
plt.show()
