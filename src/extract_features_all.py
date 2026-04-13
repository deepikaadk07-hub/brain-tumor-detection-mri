import os
import nibabel as nib
import numpy as np
import pandas as pd

# ----------------------------
# Dataset Path
# ----------------------------
root_dir = "data/MICCAI_BraTS2020_TrainingData"

patients = os.listdir(root_dir)

feature_list = []

# ----------------------------
# Loop through all patients
# ----------------------------
for patient in patients:
    try:
        patient_path = os.path.join(root_dir, patient)

        seg_path = os.path.join(patient_path, f"{patient}_seg.nii")

        # Skip if segmentation file not found
        if not os.path.exists(seg_path):
            continue

        print(f"\nProcessing: {patient}")

        # ----------------------------
        # Load Ground Truth Mask
        # ----------------------------
        seg = nib.load(seg_path).get_fdata()

        # Convert to binary (tumor = 1)
        tumor_mask = (seg > 0).astype(np.float32)

        # ----------------------------
        # Feature Extraction
        # ----------------------------
        tumor_voxels = np.sum(tumor_mask)

        slice_areas = [
            np.sum(tumor_mask[:, :, i])
            for i in range(tumor_mask.shape[2])
        ]

        affected_slices = sum(1 for area in slice_areas if area > 0)
        max_area = max(slice_areas)

        # ----------------------------
        # Labeling (based on volume)
        # ----------------------------
        if tumor_voxels < 50000:
            label = "Small"
        elif tumor_voxels < 150000:
            label = "Medium"
        else:
            label = "Large"

        # Store features
        feature_list.append([
            tumor_voxels,
            affected_slices,
            max_area,
            label
        ])

    except Exception as e:
        print(f"Skipping {patient} due to error:", e)

# ----------------------------
# Save dataset
# ----------------------------
df = pd.DataFrame(
    feature_list,
    columns=["volume", "slices", "max_area", "label"]
)

df.to_csv("tumor_features.csv", index=False)

print("\n✅ Dataset created from ALL patients!")
print(df.head())