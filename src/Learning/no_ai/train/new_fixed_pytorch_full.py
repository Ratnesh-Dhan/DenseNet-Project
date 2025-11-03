# pip install torch torchvision torchaudio
# pip install opencv-python matplotlib scikit-learn tqdm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

# ====================== CONFIG ======================
NAME = "ssd_model"
ROOT_DIR = "/mnt/d/Code/DenseNet-Project/Datasets/Traffic_Dataset/"
CLASSES_FILE = os.path.join(ROOT_DIR, "classes.txt")
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = f"/mnt/d/Code/DenseNet-Project/src/Learning/no_ai/models/{NAME}.pth"
RESULTS_DIR = "/mnt/d/Code/DenseNet-Project/src/Learning/no_ai/results"
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

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
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)  # background = 0

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        # Convert to tensor (SSD will normalize internally)
        img = F.to_tensor(img)
        
        return img, target

# ====================== MODEL ======================
def get_model(num_classes):
    model = ssd300_vgg16(weights="DEFAULT")
    
    # Properly rebuild the classification head
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]
    
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model

# ====================== UTILS ======================
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# ====================== TRAIN ======================
def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    model.to(DEVICE)
    
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        # Validation (keep in train mode to get losses)
        model.train()  # Keep in train mode for loss calculation
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_running_loss += losses.item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label="Training Loss", marker="o")
    plt.plot(val_loss_list, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Curves")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "training_curve.png"))
    plt.close()

    return model

# ====================== EVALUATION ======================
def evaluate(model, dataloader, num_classes):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Filter predictions by score
                scores = output["scores"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                
                high_score_idx = scores > SCORE_THRESHOLD
                pred_boxes = pred_boxes[high_score_idx]
                pred_labels = pred_labels[high_score_idx]
                
                true_boxes = target["boxes"].numpy()
                true_labels = target["labels"].numpy()
                
                # Match predictions to ground truth using IoU
                for gt_box, gt_label in zip(true_boxes, true_labels):
                    best_iou = 0
                    best_pred_label = 0
                    
                    for pred_box, pred_label in zip(pred_boxes, pred_labels):
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_label = pred_label
                    
                    if best_iou > IOU_THRESHOLD:
                        y_true.append(gt_label)
                        y_pred.append(best_pred_label)
                    else:
                        # No matching prediction (false negative)
                        y_true.append(gt_label)
                        y_pred.append(0)  # background class

    if len(y_true) == 0:
        print("⚠️ No ground-truth labels found for evaluation.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    acc = accuracy_score(y_true, y_pred) * 100

    print(f"\nAccuracy: {acc:.2f}%")
    print(f"Total samples evaluated: {len(y_true)}")

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

# ====================== MAIN ======================
if __name__ == "__main__":
    train_dataset = YoloDataset("train", ROOT_DIR, CLASSES_FILE)
    val_dataset = YoloDataset("val", ROOT_DIR, CLASSES_FILE)
    test_dataset = YoloDataset("test", ROOT_DIR, CLASSES_FILE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = len(train_dataset.classes) + 1  # +1 for background
    print(f"Training with {num_classes} classes (including background)")
    
    model = get_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    trained_model = train_model(model, train_loader, val_loader, optimizer, NUM_EPOCHS)
    
    # Save model
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(trained_model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved to {SAVE_PATH}")

    print("\n🔍 Evaluating on Validation set...")
    evaluate(trained_model, val_loader, num_classes)

    print("\n🔍 Evaluating on Test set...")
    evaluate(trained_model, test_loader, num_classes)