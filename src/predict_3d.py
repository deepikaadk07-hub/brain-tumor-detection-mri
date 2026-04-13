import torch
import nibabel as nib
import numpy as np
from models.unet_model import SimpleUNet
import joblib
import pandas as pd

# Load model
model = SimpleUNet()
model.load_state_dict(torch.load("models/tumor_model.pth"))
model.eval()

# Load MRI
mri_path = "data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_016/BraTS20_Training_016_flair.nii"
mri = nib.load(mri_path).get_fdata()

predicted_volume = []

for i in range(mri.shape[2]):

    slice_img = mri[:, :, i]

    # Skip empty slices
    if np.std(slice_img) == 0:
        predicted_volume.append(np.zeros_like(slice_img))
        continue

    # Normalize
    slice_img = (slice_img - np.mean(slice_img)) / np.std(slice_img)

    input_tensor = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    pred = torch.sigmoid(output)
    pred = (pred > 0.5).float()

    predicted_volume.append(pred.squeeze().numpy())

# Convert to 3D array
predicted_volume = np.stack(predicted_volume, axis=2)

print("3D Prediction shape:", predicted_volume.shape)

# ----------------------------
# Feature Extraction
# ----------------------------

voxel_volume = 1  # assume 1 mm³ (can refine later)

# Total tumor volume
tumor_voxels = np.sum(predicted_volume)
total_volume = tumor_voxels * voxel_volume

# Area per slice
slice_areas = [np.sum(predicted_volume[:, :, i]) for i in range(predicted_volume.shape[2])]

# Number of affected slices
affected_slices = sum(1 for area in slice_areas if area > 0)

print("\n===== Tumor Features =====")
print(f"Total Tumor Volume: {total_volume}")
print(f"Number of Affected Slices: {affected_slices}")
print(f"Max Area in a Slice: {max(slice_areas)}")

# Load severity model
severity_model = joblib.load("severity_model.pkl")

features = pd.DataFrame([{
    "volume": total_volume,
    "slices": affected_slices,
    "max_area": max(slice_areas)
}])

prediction = severity_model.predict(features)

print("\n===== Tumor Severity =====")
print("Predicted Category:", prediction[0])