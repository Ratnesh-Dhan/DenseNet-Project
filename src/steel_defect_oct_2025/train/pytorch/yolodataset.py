import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

# ====================== DATASET ======================
class YoloDataset(Dataset):
    def __init__(self, split, root_dir, classes_file):
        self.img_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)
        with open(classes_file) as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.imgs = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) != 5:
                        continue
                    cls, x, y, bw, bh = map(float, vals)
                    cls = int(cls)
                    
                    x1 = max(0, (x - bw / 2) * w)
                    y1 = max(0, (y - bh / 2) * h)
                    x2 = min(w, (x + bw / 2) * w)
                    y2 = min(h, (y + bh / 2) * h)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)  # +1 for background

        # Convert to tensor BEFORE resizing
        img_tensor = F.to_tensor(img)
        
        # FIX 1: Resize to 300x300 (SSD300 requirement)
        img_tensor = F.resize(img_tensor, [300, 300])
        
        # FIX 2: Scale boxes to new dimensions
        scale_x = 300 / w
        scale_y = 300 / h
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # FIX 3: Normalize with ImageNet stats
        img_tensor = F.normalize(img_tensor, 
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])

        target = {"boxes": boxes, "labels": labels}
        return img_tensor, target

