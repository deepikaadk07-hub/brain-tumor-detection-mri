import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, root_dir, slice_index=80):
        self.root_dir = root_dir
        self.patients = os.listdir(root_dir)
        self.slice_index = slice_index

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_path = os.path.join(self.root_dir, patient)

        flair = nib.load(os.path.join(patient_path, f"{patient}_flair.nii")).get_fdata()
        seg = nib.load(os.path.join(patient_path, f"{patient}_seg.nii")).get_fdata()

        img = flair[:, :, self.slice_index]
        mask = seg[:, :, self.slice_index]

        img = (img - np.mean(img)) / np.std(img)
        mask = (mask > 0).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask
