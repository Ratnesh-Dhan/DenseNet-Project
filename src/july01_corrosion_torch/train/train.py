from torch.utils.data import DataLoader
import torchvision.transforms as T
from custom_dataset_loader import CustomDataset
from load_pretrained_mask_rcnn import get_model_instance_segmentation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import os, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt

def get_transform():
    return T.Compose([T.ToTensor()])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
base_path = r"D:\NML 2nd working directory\corrosion sample piece\FINALDATASET"
# Load dataset
dataset = CustomDataset(os.path.join(base_path, "train/images"), os.path.join(base_path, "train/annotations"), transforms=get_transform())
dataset_test = CustomDataset(os.path.join(base_path, "val/images"), os.path.join(base_path, "val/annotations"), transforms=get_transform())

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load model
num_classes = 3  # Background + 1 object
model = get_model_instance_segmentation(num_classes)
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


# Training loop
num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()
#     i = 0
#     for images, targets in data_loader:
#         images = list(img.to(device) for img in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         i += 1
#         print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}], Loss: {losses.item():.4f}")
train_loss_list = []
val_loss_list = []
patience = 3
counter = 0

model_name = "mask_rcnn_july01"
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for step, (images, targets) in enumerate(data_loader, 1):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_train_loss += losses.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}], Loss: {losses.item():.4f}")

    avg_train_loss = epoch_train_loss / len(data_loader)
    train_loss_list.append(avg_train_loss)
    print(f"[Epoch {epoch+1}] Avg Training Loss: {avg_train_loss:.4f}")

    # Optional: validation phase
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Manually calculate loss during eval mode
            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):  # Ensure it's loss, not predictions
                losses = sum(loss for loss in loss_dict.values())
                epoch_val_loss += losses.item()

            # loss_dict = model(images, targets)
            # losses = sum(loss for loss in loss_dict.values())
            # epoch_val_loss += losses.item()

    avg_val_loss = epoch_val_loss / len(data_loader_test)
    val_loss_list.append(avg_val_loss)
    print(f"[Epoch {epoch+1}] Avg Validation Loss: {avg_val_loss:.4f}")

    # Compare validation loss to the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), f"../models/{model_name}_best_model_epoch_{epoch+1}.pth")
        print("✅ Saved best model with lower validation loss.")
    else:
        counter += 1
        print(f"⚠️ No improvement. Early stopping counter: {counter}/{patience}")
        if counter >= patience:
            print("⛔ Early stopping triggered.")
            break

# Plot the Loss Graphs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
if not os.path.exists("../results"):
    os.makedirs("../results")
plt.savefig(f"../results/{model_name}_loss_plot_epoch_{num_epochs}.png")  # Saves to file
plt.show()                    # Displays in notebook or window

loss_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_loss_list,
    'Validation Loss': val_loss_list
})
loss_df.to_csv(f"../results/{model_name}_loss_log_epoch_{num_epochs}.csv", index=False)

all_preds = []
all_targets = []

model.eval()
with torch.no_grad():
    for images, targets in data_loader_test:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for i in range(len(outputs)):
            pred_labels = outputs[i]['labels'].cpu().numpy()
            true_labels = targets[i]['labels'].cpu().numpy()

            # Optionally match length by min length if needed
            min_len = min(len(pred_labels), len(true_labels))
            all_preds.extend(pred_labels[:min_len])
            all_targets.extend(true_labels[:min_len])

# Define class names (no background)
class_names = ['corrosion', 'piece']

# Create normalized confusion matrix
cm = confusion_matrix(all_targets, all_preds, normalize='true')  # row-normalized

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig(f'../results/{model_name}_normalized_confusion_matrix_epoch_{num_epochs}.png')
plt.close()

# Create classification report
report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

# Save classification report to file
with open(f"../results/{model_name}_classification_report_epoch_{num_epochs}.txt", "w") as f:
    f.write(report)

print(f"Confusion matrix saved as '{model_name}_normalized_confusion_matrix_epoch_{num_epochs}.png'")
print(f"Classification report saved as '{model_name}_classification_report_epoch_{num_epochs}.txt'")