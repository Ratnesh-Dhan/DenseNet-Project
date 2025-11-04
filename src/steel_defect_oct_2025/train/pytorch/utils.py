import os, matplotlib.pyplot as plt, numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
import torch
from tqdm import tqdm

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

# ====================== PLOT TRAINING CURVES ======================
def plot_training_curves(train_loss, val_loss, train_acc, val_acc, best_epoch, RESULTS_DIR):
    """Plot loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(train_loss) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to {RESULTS_DIR}/training_curves.png")

# ====================== CALCULATE ACCURACY ======================
def calculate_accuracy(model, dataloader, DEVICE, SCORE_THRESHOLD, max_batches=None):
    """Calculate accuracy on a subset of data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
                
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                true_labels = target["labels"].numpy()
                
                high_score_idx = scores > SCORE_THRESHOLD
                pred_labels = pred_labels[high_score_idx]
                
                if len(pred_labels) > 0 and len(true_labels) > 0:
                    # Simple accuracy: do predicted classes match any true class?
                    for true_label in true_labels:
                        if true_label in pred_labels:
                            correct += 1
                    total += len(true_labels)
    
    model.train()
    return (correct / total * 100) if total > 0 else 0

# ====================== PERCENTAGE CONFUSION MATRIX ======================
def plot_percentage_confusion_matrix(y_true, y_pred, class_names, acc, prec, recall, f1, split_name, RESULTS_DIR):
    """Plot confusion matrix with percentages"""
    # Get unique labels (excluding background)
    labels = sorted(list(set(y_true)))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Convert to percentages (row-wise)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percent = np.nan_to_num(cm_percent)  # Replace NaN with 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot heatmap
    im = ax.imshow(cm_percent, cmap='Blues', interpolation='nearest', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=12)
    
    # Set ticks and labels
    label_names = [class_names[l-1] if l > 0 and l <= len(class_names) else f"Class {l}" 
                   for l in labels]
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(label_names, fontsize=10)
    
    # Add text annotations
    thresh = 50
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text = f'{percent:.1f}%\n({count})'
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                   color="white" if percent > thresh else "black",
                   fontweight='bold' if i == j else 'normal')
    
    # Labels and title
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    title = f'{split_name} Set Confusion Matrix (Percentage)\n'
    title += f'Acc: {acc:.1f}% | Prec: {prec:.1f}% | Recall: {recall:.1f}% | F1: {f1:.1f}%'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = f"confusion_matrix_percent_{split_name.lower()}.png"
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {RESULTS_DIR}/{filename}")

# ====================== CLASSIFICATION REPORT ======================
def print_classification_report(y_true, y_pred, class_names, split_name, RESULTS_DIR):
    """Print and save detailed classification report"""
    labels = sorted(list(set(y_true)))
    label_names = [class_names[l-1] if l > 0 and l <= len(class_names) else f"Class {l}" 
                   for l in labels]
    
    report = classification_report(y_true, y_pred, labels=labels, 
                                   target_names=label_names, digits=3)
    
    print(f"\n{'='*70}")
    print(f"{split_name.upper()} SET - CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(report)
    
    # Save to file
    report_path = os.path.join(RESULTS_DIR, f"classification_report_{split_name.lower()}.txt")
    with open(report_path, 'w') as f:
        f.write(f"{split_name.upper()} SET - CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(report)
    
    print(f"✅ Classification report saved to {report_path}")

# ====================== COMPREHENSIVE EVALUATION ======================
def evaluate_comprehensive(DEVICE, SCORE_THRESHOLD, IOU_THRESHOLD, model, dataloader, num_classes, class_names, split_name="Test"):
    """Complete evaluation with confusion matrix, classification report, and metrics"""
    model.eval()
    y_true, y_pred = [], []
    total_gt = 0
    total_pred = 0
    matched = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Evaluating {split_name}"):
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
        print(f"⚠️ No predictions matched ground truth in {split_name} set!")
        return

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    recall = matched / total_gt * 100 if total_gt > 0 else 0
    precision = matched / total_pred * 100 if total_pred > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*70}")
    print(f"{split_name.upper()} SET RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}% ({matched}/{total_pred})")
    print(f"Recall: {recall:.2f}% ({matched}/{total_gt})")
    print(f"F1 Score: {f1:.2f}%")
    print(f"{'='*70}\n")

    # Plot percentage confusion matrix
    plot_percentage_confusion_matrix(y_true, y_pred, class_names, accuracy, 
                                     precision, recall, f1, split_name)
    
    # Print classification report
    print_classification_report(y_true, y_pred, class_names, split_name)