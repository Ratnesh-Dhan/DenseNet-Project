# pip install torch torchvision torchaudio
# pip install opencv-python matplotlib scikit-learn tqdm

import os, numpy as np, matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from tqdm import tqdm
from utils import evaluate_comprehensive, calculate_iou, inspect_dataset, plot_training_curves, calculate_accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
# from yolodataset import YoloDataset
from xmldataset import XMLDataset
from model import get_model
from earlystopping import EarlyStopping

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
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# ====================== TRAIN ======================
def train_model(model, train_loader, val_loader, optimizer, num_epochs):
    model.to(DEVICE)
    
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )

    # Adding early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.01)
    best_val_loss = float('inf')
    best_epoch = 0

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

        # Calculate training accuracy
        train_acc = calculate_accuracy(model, train_loader, DEVICE, SCORE_THRESHOLD)
        train_acc_list.append(train_acc)
        
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

        # Calculate validation accuracy
        val_acc = calculate_accuracy(model, val_loader, DEVICE, SCORE_THRESHOLD)
        val_acc_list.append(val_acc)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save best model checkpoint
            torch.save(model.state_dict(), SAVE_PATH.replace(".pth", "_best.pth"))
            print(f"✅ New best model saved! (Val Loss: {avg_val_loss:.4f})")
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
             f"Val Loss: {avg_val_loss:.4f} {'⭐ BEST' if epoch+1 == best_epoch else ''}")

        # Check early stopping
        if early_stopping(avg_val_loss, model):
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"📊 Best validation loss: {early_stopping.best_loss:.4f}")
            # Restore best model
            model.load_state_dict(early_stopping.best_model)
            break

    print(f"\n🏆 Best model was at epoch {best_epoch} with val loss: {best_val_loss:.4f}")
        
    plot_training_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list, best_epoch, RESULTS_DIR)   
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
    train_dataset = XMLDataset("train", ROOT_DIR, CLASSES_FILE)
    val_dataset = XMLDataset("val", ROOT_DIR, CLASSES_FILE)
    test_dataset = XMLDataset("test", ROOT_DIR, CLASSES_FILE)

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

    # print("\n🔍 Evaluating on Validation set...")
    # evaluate(trained_model, val_loader, num_classes, train_dataset.classes)

    # print("\n🔍 Evaluating on Test set...")
    # evaluate(trained_model, test_loader, num_classes, train_dataset.classes)
        # Comprehensive evaluation on validation set
    print("\n" + "="*70)
    print("EVALUATING ON VALIDATION SET")
    print("="*70)
    evaluate_comprehensive(DEVICE, SCORE_THRESHOLD, IOU_THRESHOLD, trained_model, val_loader, num_classes, 
                          train_dataset.classes, "Validation")

    # Comprehensive evaluation on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    evaluate_comprehensive(DEVICE, SCORE_THRESHOLD, IOU_THRESHOLD, trained_model, test_loader, num_classes, 
                          train_dataset.classes, "Test")