import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_root_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_root_dir = mask_root_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(image_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        mask_dir = os.path.join(self.mask_root_dir, self.image_files[idx].split('.')[0])
        mask_files = sorted(os.listdir(mask_dir))

        masks = []
        labels = []
        boxes = []

        for mask_file in mask_files:
            mask = Image.open(os.path.join(mask_dir, mask_file)).convert("L")
            mask = np.array(mask)
            mask = (mask > 0).astype(np.uint8)  # Make binary

            if mask.max() == 0:
                continue  # Skip empty masks

            masks.append(mask)

            # Label assignment
            if "corrosion" in mask_file.lower():
                labels.append(1)
            elif "piece_1" in mask_file.lower():
                labels.append(2)
            else:
                raise ValueError(f"Unknown class: {mask_file}")

            # Bounding box
            pos = np.argwhere(mask)
            ymin, xmin = pos.min(axis=0)
            ymax, xmax = pos.max(axis=0)
            if xmax <= xmin or ymax <= ymin:
                continue  # Skip invalid boxes
            boxes.append([xmin, ymin, xmax, ymax])

        if len(masks) == 0:
            raise ValueError(f"No valid masks for image {self.image_files[idx]}")

        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)
        
        # # this is for debugging
        # plt.imshow(img.permute(1, 2, 0))  # if img is tensor after transform
        # plt.imshow(masks[0], alpha=0.5)
        # plt.title(f"Labels: {labels.tolist()}")
        # plt.show()

        return img, target


    def __len__(self):
        return len(self.image_files)
