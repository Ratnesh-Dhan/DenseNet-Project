"""
optimizer_comparison.py

Train SSD model with different optimizers and compare results
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from sklearn.metrics import accuracy_score
from xmldataset import XMLDataset
from utils import evaluate_comprehensive

# ====================== CONFIG ======================
ROOT_DIR = "../../../../Datasets/NEU-DET/"
CLASSES_FILE = os.path.join(ROOT_DIR, "classes.txt")
DATASET_FORMAT = "yolo"  # or "xml"
BATCH_SIZE = 8
NUM_EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "../results/optimizer_comparison"
SCORE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

os.makedirs(RESULTS_DIR, exist_ok=True)

# ====================== OPTIMIZER CONFIGURATIONS ======================
OPTIMIZER_CONFIGS = {
    'Adam': {
        'optimizer': optim.Adam,
        'params': {'lr': 0.0005, 'betas': (0.9, 0.999), 'weight_decay': 0.0001},
        'color': 'blue'
    },
    'AdamW': {
        'optimizer': optim.AdamW,
        'params': {'lr': 0.0005, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
        'color': 'green'
    },
    'SGD': {
        'optimizer': optim.SGD,
        'params': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0001},
        'color': 'red'
    },
    'SGD_Nesterov': {
        'optimizer': optim.SGD,
        'params': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0001, 'nesterov': True},
        'color': 'orange'
    },
    'RMSprop': {
        'optimizer': optim.RMSprop,
        'params': {'lr': 0.0005, 'alpha': 0.99, 'weight_decay': 0.0001},
        'color': 'purple'
    },
    'Adagrad': {
        'optimizer': optim.Adagrad,
        'params': {'lr': 0.001, 'weight_decay': 0.0001},
        'color': 'brown'
    }
}

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

# ====================== TRAINING ======================
def train_with_optimizer(model, train_loader, val_loader, optimizer, 
                         optimizer_name, num_epochs):
    """Train model with specific optimizer and return metrics"""
    model.to(DEVICE)
    
    train_loss_history = []
    val_loss_history = []
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*70}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
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
        train_loss_history.append(avg_train_loss)
        
        # Validation
        model.train()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_running_loss += losses.item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save best model
            save_path = os.path.join(RESULTS_DIR, f"best_model_{optimizer_name}.pth")
            torch.save(model.state_dict(), save_path)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} "
              f"{'⭐' if epoch+1 == best_epoch else ''}")
    
    print(f"\n✅ Best epoch: {best_epoch} with val loss: {best_val_loss:.4f}")
    
    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }

# ====================== EVALUATION METRICS ======================
def evaluate_model(model, test_loader, num_classes, class_names):
    """Quick evaluation to get accuracy, precision, recall"""
    model.eval()
    y_true, y_pred = [], []
    total_gt = 0
    total_pred = 0
    matched = 0

    with torch.no_grad():
        for images, targets in test_loader:
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
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    accuracy = accuracy_score(y_true, y_pred) * 100
    recall = matched / total_gt * 100 if total_gt > 0 else 0
    precision = matched / total_pred * 100 if total_pred > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ====================== PLOT COMPARISON ======================
def plot_optimizer_comparison(results):
    """Plot comparison of all optimizers"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for opt_name, metrics in results.items():
        config = OPTIMIZER_CONFIGS[opt_name]
        ax.plot(metrics['train_loss'], label=opt_name, 
               color=config['color'], marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for opt_name, metrics in results.items():
        config = OPTIMIZER_CONFIGS[opt_name]
        ax.plot(metrics['val_loss'], label=opt_name, 
               color=config['color'], marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best Validation Loss (Bar chart)
    ax = axes[1, 0]
    opt_names = list(results.keys())
    best_losses = [results[opt]['best_val_loss'] for opt in opt_names]
    colors = [OPTIMIZER_CONFIGS[opt]['color'] for opt in opt_names]
    bars = ax.bar(opt_names, best_losses, color=colors, alpha=0.7)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Best Validation Loss by Optimizer', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, loss in zip(bars, best_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{loss:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Performance Metrics (if available)
    ax = axes[1, 1]
    if 'test_metrics' in results[list(results.keys())[0]]:
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        x = np.arange(len(metrics_names))
        width = 0.15
        
        for i, opt_name in enumerate(opt_names):
            test_metrics = results[opt_name]['test_metrics']
            values = [test_metrics['accuracy'], test_metrics['precision'], 
                     test_metrics['recall'], test_metrics['f1']]
            offset = width * (i - len(opt_names)/2)
            ax.bar(x + offset, values, width, label=opt_name, 
                  color=OPTIMIZER_CONFIGS[opt_name]['color'], alpha=0.7)
        
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Test metrics not available', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'optimizer_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Comparison plot saved to {save_path}")

# ====================== SAVE RESULTS TABLE ======================
def save_results_table(results):
    """Save results as text table"""
    table_path = os.path.join(RESULTS_DIR, 'optimizer_results.txt')
    
    with open(table_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("OPTIMIZER COMPARISON RESULTS\n")
        f.write("="*100 + "\n\n")
        
        # Header
        f.write(f"{'Optimizer':<20} {'Best Val Loss':<15} {'Best Epoch':<12}")
        if 'test_metrics' in results[list(results.keys())[0]]:
            f.write(f"{'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        else:
            f.write("\n")
        f.write("-"*100 + "\n")
        
        # Data rows
        for opt_name in sorted(results.keys(), key=lambda x: results[x]['best_val_loss']):
            metrics = results[opt_name]
            f.write(f"{opt_name:<20} {metrics['best_val_loss']:<15.4f} {metrics['best_epoch']:<12}")
            
            if 'test_metrics' in metrics:
                test = metrics['test_metrics']
                f.write(f"{test['accuracy']:<12.2f} {test['precision']:<12.2f} "
                       f"{test['recall']:<12.2f} {test['f1']:<12.2f}\n")
            else:
                f.write("\n")
        
        f.write("="*100 + "\n\n")
        
        # Optimizer parameters
        f.write("OPTIMIZER PARAMETERS:\n")
        f.write("-"*100 + "\n")
        for opt_name, config in OPTIMIZER_CONFIGS.items():
            f.write(f"\n{opt_name}:\n")
            for param, value in config['params'].items():
                f.write(f"  {param}: {value}\n")
    
    print(f"✅ Results table saved to {table_path}")

# ====================== MAIN ======================
if __name__ == "__main__":
    print("="*70)
    print("SSD OPTIMIZER COMPARISON")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = XMLDataset("train", ROOT_DIR, CLASSES_FILE)
    val_dataset = XMLDataset("val", ROOT_DIR, CLASSES_FILE)
    test_dataset = XMLDataset("test", ROOT_DIR, CLASSES_FILE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    num_classes = len(train_dataset.classes) + 1
    print(f"Classes: {train_dataset.classes}")
    print(f"Device: {DEVICE}\n")
    
    # Store results for all optimizers
    all_results = {}
    
    # Train with each optimizer
    for opt_name, config in OPTIMIZER_CONFIGS.items():
        print(f"\n{'#'*70}")
        print(f"TESTING OPTIMIZER: {opt_name}")
        print(f"Parameters: {config['params']}")
        print(f"{'#'*70}")
        
        # Create fresh model
        model = get_model(num_classes)
        
        # Create optimizer
        optimizer = config['optimizer'](model.parameters(), **config['params'])
        
        # Train
        train_results = train_with_optimizer(
            model, train_loader, val_loader, optimizer, opt_name, NUM_EPOCHS
        )
        
        # Load best model for evaluation
        best_model_path = os.path.join(RESULTS_DIR, f"best_model_{opt_name}.pth")
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate on test set
        print(f"\nEvaluating {opt_name} on test set...")
        test_metrics = evaluate_model(model, test_loader, num_classes, train_dataset.classes)
        
        print(f"\nTest Results for {opt_name}:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.2f}%")
        print(f"  Precision: {test_metrics['precision']:.2f}%")
        print(f"  Recall:    {test_metrics['recall']:.2f}%")
        print(f"  F1 Score:  {test_metrics['f1']:.2f}%")
        
        # Store results
        train_results['test_metrics'] = test_metrics
        all_results[opt_name] = train_results
    
    # Plot comparison
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    plot_optimizer_comparison(all_results)
    
    # Save results table
    save_results_table(all_results)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Find best optimizer
    best_opt = min(all_results.keys(), key=lambda x: all_results[x]['best_val_loss'])
    best_test_opt = max(all_results.keys(), key=lambda x: all_results[x]['test_metrics']['f1'])
    
    print(f"\n🏆 Best Validation Loss: {best_opt}")
    print(f"   Loss: {all_results[best_opt]['best_val_loss']:.4f}")
    
    print(f"\n🏆 Best Test Performance: {best_test_opt}")
    print(f"   F1 Score: {all_results[best_test_opt]['test_metrics']['f1']:.2f}%")
    
    print(f"\n📁 All results saved to: {RESULTS_DIR}/")
    print("="*70)