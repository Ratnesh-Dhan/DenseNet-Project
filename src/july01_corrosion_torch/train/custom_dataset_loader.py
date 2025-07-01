import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

class CorrosionDataset(torch.utils.data.Dataset):
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

        for mask_file in mask_files:
            mask = Image.open(os.path.join(mask_dir, mask_file)).convert("L")
            mask = np.array(mask)
            if mask.max() == 0:
                continue  # Skip empty masks

            masks.append(mask)

            if "corrosion" in mask_file.lower():
                labels.append(1)
            elif "piece_1" in mask_file.lower():
                labels.append(2)
            else:
                raise ValueError(f"Unknown class: {mask_file}")

        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        boxes = []
        for m in masks:
            pos = m.nonzero()
            xmin = pos[:, 1].min()
            xmax = pos[:, 1].max()
            ymin = pos[:, 0].min()
            ymax = pos[:, 0].max()
            boxes.append([xmin, ymin, xmax, ymax])
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
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_files)
