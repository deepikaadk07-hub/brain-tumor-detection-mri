import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models.unet_model import SimpleUNet

# ----------------------------
# Load Model
# ----------------------------
model = SimpleUNet()
model.load_state_dict(torch.load("models/tumor_model.pth"))
model.eval()

# ----------------------------
# Load MRI and Ground Truth
# ----------------------------
base_path = "data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_016/"

mri = nib.load(base_path + "BraTS20_Training_016_flair.nii").get_fdata()
gt = nib.load(base_path + "BraTS20_Training_016_seg.nii").get_fdata()

# Select slice
slice_index = 80

slice_img = mri[:, :, slice_index]
gt_slice = gt[:, :, slice_index]

# Skip empty slice
if np.std(slice_img) == 0:
    raise ValueError("Selected slice is empty. Choose another slice.")

# Normalize MRI
slice_img = (slice_img - np.mean(slice_img)) / np.std(slice_img)

# Convert GT to binary (tumor = 1, background = 0)
gt_slice = (gt_slice > 0).astype(np.float32)

# Convert to tensors
input_tensor = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
ground_truth_mask = torch.tensor(gt_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ----------------------------
# Predict
# ----------------------------
with torch.no_grad():
    output = model(input_tensor)

probability_map = torch.sigmoid(output)
threshold = 0.5
prediction = (probability_map > threshold).float()

pred_mask = prediction.squeeze().numpy()
gt_mask = ground_truth_mask.squeeze().numpy()

# ----------------------------
# FIGURE 1 — Overlay (Heatmap)
# ----------------------------
plt.figure(figsize=(6,6))
plt.imshow(slice_img, cmap="gray")
plt.imshow(pred_mask, cmap="jet", alpha=0.5)
plt.title("Tumor Detection Overlay (Heatmap)")
plt.axis("off")
plt.show()

# ----------------------------
# FIGURE 2 — Comparison
# ----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.imshow(slice_img, cmap="gray")
plt.title("Original MRI")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(gt_mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.show()

# ----------------------------
# Evaluation Metrics
# ----------------------------
def dice_score(pred, target, smooth=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    correct = (pred == target).sum()
    total = target.numel()
    return correct.float() / total

dice = dice_score(prediction, ground_truth_mask)
iou = iou_score(prediction, ground_truth_mask)
acc = pixel_accuracy(prediction, ground_truth_mask)

print("\n===== Evaluation Metrics =====")
print(f"Dice Score: {dice:.4f}")
print(f"IoU Score: {iou:.4f}")
print(f"Pixel Accuracy: {acc:.4f}")

# ----------------------------
# Confusion Matrix
# ----------------------------
pred_np = prediction.squeeze().numpy().flatten()
gt_np = ground_truth_mask.squeeze().numpy().flatten()

cm = confusion_matrix(gt_np, pred_np)
print("\nConfusion Matrix:")
print(cm)

# ----------------------------
# Tumor Area Percentage
# ----------------------------
tumor_pixels = prediction.sum()
total_pixels = prediction.numel()

tumor_percentage = (tumor_pixels / total_pixels) * 100

print(f"\nTumor Area Percentage: {tumor_percentage:.2f}%")
