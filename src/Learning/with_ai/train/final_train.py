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
NAME = "ssd_model_fixed"
ROOT_DIR = "../../../../Datasets/Traffic_Dataset/"
CLASSES_FILE = os.path.join(ROOT_DIR, "classes.txt")
BATCH_SIZE = 8  # Increased from 4
NUM_EPOCHS = 25  # Increased from 10
LR = 0.0005  # Reduced from 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = f"../models/{NAME}.pth"
RESULTS_DIR = "../results"
SCORE_THRESHOLD = 0.3  # Reduced from 0.5
IOU_THRESHOLD = 0.5

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

# ====================== MODEL ======================
def get_model(num_classes):
    model = ssd300_vgg16(weights="DEFAULT")
    
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
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def inspect_dataset(dataset, num_samples=3):
    print(f"\n{'='*50}")
    print(f"Dataset Inspection ({dataset.img_dir})")
    print(f"{'='*50}")
    print(f"Total images: {len(dataset)}")
    print(f"Classes ({len(dataset.classes)}): {dataset.classes}")
    
    for i in range(min(num_samples, len(dataset))):
        img, target = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Num boxes: {len(target['boxes'])}")
        print(f"  Labels: {target['labels'].tolist()}")
    
    print(f"{'='*50}\n")

# ====================== TRAIN ======================
def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    model.to(DEVICE)
    
    train_loss_list = []
    val_loss_list = []
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += losses.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        # Validation
        model.train()
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
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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
def evaluate(model, dataloader, num_classes, class_names):
    model.eval()
    y_true, y_pred = [], []
    total_gt = 0
    total_pred = 0
    matched = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                scores = output["scores"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                
                high_score_idx = scores > SCORE_THRESHOLD
                pred_boxes = pred_boxes[high_score_idx]
                pred_labels = pred_labels[high_score_idx]
                
                true_boxes = target["boxes"].numpy()
                true_labels = target["labels"].numpy()
                
                total_gt += len(true_labels)
                total_pred += len(pred_labels)
                
                matched_gt = set()
                matched_pred = set()
                
                for i, (gt_box, gt_label) in enumerate(zip(true_boxes, true_labels)):
                    best_iou = 0
                    best_pred_idx = -1
                    
                    for j, pred_box in enumerate(pred_boxes):
                        if j in matched_pred:
                            continue
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = j
                    
                    if best_iou > IOU_THRESHOLD and best_pred_idx != -1:
                        matched_gt.add(i)
                        matched_pred.add(best_pred_idx)
                        y_true.append(gt_label)
                        y_pred.append(pred_labels[best_pred_idx])
                        matched += 1

    if len(y_true) == 0:
        print("⚠️ No predictions matched ground truth!")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, num_classes)))
    acc = accuracy_score(y_true, y_pred) * 100
    recall = matched / total_gt * 100 if total_gt > 0 else 0
    precision = matched / total_pred * 100 if total_pred > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*50}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {precision:.2f}% ({matched}/{total_pred})")
    print(f"Recall: {recall:.2f}% ({matched}/{total_gt})")
    print(f"F1 Score: {f1:.2f}%")
    print(f"{'='*50}\n")

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues", interpolation='nearest')
    plt.title(f"Confusion Matrix\nAcc: {acc:.1f}% | Prec: {precision:.1f}% | Recall: {recall:.1f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    
    tick_marks = np.arange(num_classes - 1)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

# ====================== MAIN ======================
if __name__ == "__main__":
    print("Initializing datasets...")
    train_dataset = YoloDataset("train", ROOT_DIR, CLASSES_FILE)
    val_dataset = YoloDataset("val", ROOT_DIR, CLASSES_FILE)
    test_dataset = YoloDataset("test", ROOT_DIR, CLASSES_FILE)

    inspect_dataset(train_dataset, num_samples=3)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = len(train_dataset.classes) + 1
    print(f"\nTraining with {num_classes} classes (including background)")
    print(f"Object classes: {train_dataset.classes}\n")
    
    model = get_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, optimizer, NUM_EPOCHS)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(trained_model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved to {SAVE_PATH}")

    print("\n🔍 Evaluating on Validation set...")
    evaluate(trained_model, val_loader, num_classes, train_dataset.classes)

    print("\n🔍 Evaluating on Test set...")
    evaluate(trained_model, test_loader, num_classes, train_dataset.classes)