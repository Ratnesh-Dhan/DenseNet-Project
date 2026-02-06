import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T
import json

# ======================
# Build COCO category remap ONCE
# ======================
ANN_FILE = "/mnt/d/DATASETS/Aluminium_ingot/merged_coco_fixed.json"

with open(ANN_FILE) as f:
    coco_json = json.load(f)

# COCO category ids are arbitrary → remap to [1, 2, ...]
cat_ids = sorted([c["id"] for c in coco_json["categories"]])
CAT_ID_MAP = {cid: i + 1 for i, cid in enumerate(cat_ids)}
NUM_CLASSES = len(cat_ids) + 1  # + background

# ======================
# Dataset
# ======================
class IngotCoco(CocoDetection):
    def __init__(self, img_dir, ann_file):
        super().__init__(img_dir, ann_file)

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(CAT_ID_MAP[ann["category_id"]])

        # ⚠ handle images with NO annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        img = T.ToTensor()(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ======================
# Data
# ======================
dataset = IngotCoco(
    img_dir="/mnt/d/DATASETS/Aluminium_ingot/images",
    ann_file=ANN_FILE
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

# ======================
# Model
# ======================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, NUM_CLASSES
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ======================
# Optimizer
# ======================
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.005, momentum=0.9, weight_decay=0.0005
)

# ======================
# Train loop
# ======================
best_loss = float("inf")
model.train()

for epoch in range(5):
    epoch_loss = 0.0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    # ✅ SAVE BEST MODEL
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "fasterrcnn_best.pth")
        print(">> Saved best model")

print(f"✅ Best model saved with loss: {best_loss:.4f}")

# load best model
model.load_state_dict(torch.load("fasterrcnn_best.pth"))
model.eval()