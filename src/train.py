import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
from models.unet_model import SimpleUNet

# ----------------------------
# Load MRI and Mask
# ----------------------------
mri_path = "data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_016/BraTS20_Training_016_flair.nii"
mask_path = "data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_016/BraTS20_Training_016_seg.nii"

mri = nib.load(mri_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

# ----------------------------
# Model Setup
# ----------------------------
model = SimpleUNet()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# Dice Loss
# ----------------------------
def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ----------------------------
# Training Loop
# ----------------------------
epochs = 5

for epoch in range(epochs):

    total_loss = 0
    slice_count = 0

    for slice_index in range(mri.shape[2]):

        img_slice = mri[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]

        # Skip empty MRI slices
        if np.std(img_slice) == 0:
            continue

        # Convert mask to binary (VERY IMPORTANT)
        mask_slice = (mask_slice > 0).astype(np.float32)

        # Skip slices without tumor
        if np.sum(mask_slice) == 0:
            continue

        # Normalize image
        img_slice = (img_slice - np.mean(img_slice)) / np.std(img_slice)

        # Convert to tensors
        input_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Forward
        output = model(input_tensor)

        # Loss
        loss = dice_loss(output, target_tensor)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        slice_count += 1

    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/slice_count:.4f}")

# ----------------------------
# Save Model
# ----------------------------
torch.save(model.state_dict(), "models/tumor_model.pth")
print("Model saved successfully.")
