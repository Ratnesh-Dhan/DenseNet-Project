import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
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
SAVE_PATH =f"/mnt/d/Code/DenseNet-Project/src/Learning/no_ai/models/{NAME}.pth"

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
                    x1 = (x - bw / 2) * w
                    y1 = (y - bh / 2) * h
                    x2 = (x + bw / 2) * w
                    y2 = (y + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)  # background = 0

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        img = F.to_tensor(img)
        return img, target

# ====================== MODEL ======================
def get_model(num_classes):
    model = ssd300_vgg16(weights="DEFAULT")
    model.head.classification_head.num_classes = num_classes
    return model

# ====================== TRAIN ======================
def train_model(model, dataloader, optimizer, num_epochs):
    model.to(DEVICE)
    model.train()

    loss_list = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        avg_loss = running_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    plt.figure()
    plt.plot(loss_list, label="Training Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig("../results/training_curve.png")
    plt.show()

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
                pred_labels = output["labels"].cpu().numpy()
                true_labels = target["labels"].numpy()
                if len(true_labels) > 0 and len(pred_labels) > 0:
                    y_true.extend(true_labels)
                    y_pred.extend(pred_labels)

    if len(y_true) == 0:
        print("⚠️ No ground-truth labels found for evaluation.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    acc = accuracy_score(y_true, y_pred) * 100

    print(f"\nAccuracy: {acc:.2f}%")

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig("../results/confusion_matrix.png")
    plt.show()

# ====================== MAIN ======================
if __name__ == "__main__":
    train_dataset = YoloDataset("train", ROOT_DIR, CLASSES_FILE)
    val_dataset = YoloDataset("val", ROOT_DIR, CLASSES_FILE)
    test_dataset = YoloDataset("test", ROOT_DIR, CLASSES_FILE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = len(train_dataset.classes) + 1  # +1 for background
    model = get_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    trained_model = train_model(model, train_loader, optimizer, NUM_EPOCHS)
    torch.save(trained_model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved to {SAVE_PATH}")

    print("\n🔍 Evaluating on Validation set...")
    evaluate(trained_model, val_loader, num_classes)

    print("\n🔍 Evaluating on Test set...")
    evaluate(trained_model, test_loader, num_classes)
