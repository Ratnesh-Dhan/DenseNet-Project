import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Custom Dataset ===
class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, classes):
        self.img_dir, self.label_dir, self.classes = img_dir, label_dir, classes
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        lbl_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))
        img = F.pil_to_tensor(Image.open(img_path).convert("RGB")) / 255.0

        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                x_min = x - w/2
                y_min = y - h/2
                x_max = x + w/2
                y_max = y + h/2
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(cls) + 1)

        boxes = torch.tensor(boxes) * img.shape[1]
        target = {"boxes": boxes, "labels": torch.tensor(labels)}
        return img, target

# === Load Model ===
num_classes = 4  # include background
model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
model.head.classification_head.num_classes = num_classes
model.train()

# === Optimizer ===
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# === DataLoader ===
train_dataset = YoloDataset("dataset/images/train", "dataset/labels/train", classes)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# === Training Loop ===
losses = []
for epoch in range(10):
    epoch_loss = 0
    for imgs, targets in train_loader:
        imgs = list(img.to("cuda") for img in imgs)
        targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")
    losses.append(epoch_loss)

# === Plot Loss Curve ===
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("ssd_loss_curve.png")
plt.show()

# === Save Model ===
torch.save(model.state_dict(), "ssd_mobilenetv3_transfer.pth")
