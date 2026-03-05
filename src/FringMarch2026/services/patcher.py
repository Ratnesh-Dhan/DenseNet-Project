import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import random

class FringeDataset(Dataset):
    def __init__(self, img_dir, height_dir, patch=128):

        self.img_dir = img_dir
        self.height_dir = height_dir
        self.files = os.listdir(img_dir)
        self.patch = patch

    def __len__(self):
        return len(self.files)

    def augment(self, img, height):
        if random.random() > 0.5:
            img = np.flip(img,1)
            height = np.flip(height,1)

        if random.random() > 0.5:
            img = np.flip(img,0)
            height = np.flip(height,0)

        return img.copy(), height.copy()

    def __getitem__(self, idx):

        file = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir,file),0)
        img = img.astype(np.float32)/255.0

        height = np.load(os.path.join(self.height_dir,file.replace(".bmp",".npy")))

        # normalize height
        height = (height - height.mean())/height.std()

        # random crop
        x = random.randint(0, img.shape[0]-self.patch)
        y = random.randint(0, img.shape[1]-self.patch)

        img_patch = img[x:x+self.patch, y:y+self.patch]

        hx = x*2
        hy = y*2
        height_patch = height[hx:hx+self.patch*2, hy:hy+self.patch*2]

        img_patch, height_patch = self.augment(img_patch, height_patch)

        img_patch = torch.tensor(img_patch).unsqueeze(0)
        height_patch = torch.tensor(height_patch).unsqueeze(0)

        return img_patch.float(), height_patch.float()