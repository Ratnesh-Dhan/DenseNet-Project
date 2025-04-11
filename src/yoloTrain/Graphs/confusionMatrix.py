import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Config
gt_dir = "../../../Datasets/yoloPCB/labels/val"  # path to ground truth labels
pred_dir = "../../../runs/detect/val5/labels"  # path to predicted labels
class_names = [
    "Cap1", "Cap2", "Cap3", "Cap4", "MOSFET", "Mov", "Resestor",
    "Resistor", "Transformer", "Ic", "Diode", "Cap6",
    "Transistor", "Potentiometer"
]
conf_threshold = 0.25  # ignore predictions below this confidence

# Mapping
resestor_idx = class_names.index("Resestor")
resistor_idx = class_names.index("Resistor")

# IOU helper
def compute_iou(box1, box2):
    # Format: [x_center, y_center, width, height] (normalized)
    x1_min = box1[0] - box1[2]/2
    y1_min = box1[1] - box1[3]/2
    x1_max = box1[0] + box1[2]/2
    y1_max = box1[1] + box1[3]/2

    x2_min = box2[0] - box2[2]/2
    y2_min = box2[1] - box2[3]/2
    x2_max = box2[0] + box2[2]/2
    y2_max = box2[1] + box2[3]/2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area != 0 else 0

# Process
y_true, y_pred = [], []

for filename in os.listdir(gt_dir):
    if not filename.endswith(".txt"):
        continue

    gt_path = os.path.join(gt_dir, filename)
    pred_path = os.path.join(pred_dir, filename)

    if not os.path.exists(pred_path):
        continue

    with open(gt_path) as f:
        gt_lines = [list(map(float, line.strip().split())) for line in f]
    with open(pred_path) as f:
        pred_lines = [list(map(float, line.strip().split())) for line in f if float(line.strip().split()[-1]) >= conf_threshold]

    gt_boxes = [[int(l[0]), l[1], l[2], l[3], l[4]] for l in gt_lines]
    pred_boxes = [[int(l[0]), l[1], l[2], l[3], l[4], l[5]] for l in pred_lines]

    used_preds = set()
    for gt in gt_boxes:
        gt_cls, *gt_box = gt
        best_iou, best_pred_idx = 0, -1

        for idx, pred in enumerate(pred_boxes):
            if idx in used_preds:
                continue
            pred_cls, *pred_box, conf = pred
            iou = compute_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou, best_pred_idx = iou, idx

        if best_iou > 0.5 and best_pred_idx != -1:
            pred_cls = pred_boxes[best_pred_idx][0]
            used_preds.add(best_pred_idx)
            y_true.append(gt_cls)
            y_pred.append(pred_cls)

# Merge Resestor → Resistor
y_true = [resistor_idx if y == resestor_idx else y for y in y_true]
y_pred = [resistor_idx if p == resestor_idx else p for p in y_pred]

# Remove Resestor from class list
exclude_indices = [resestor_idx]
include_indices = [i for i in range(len(class_names)) if i not in exclude_indices]
index_map = {old: new for new, old in enumerate(include_indices)}

# Filter and remap
filtered_y_true = [y for y, p in zip(y_true, y_pred) if y in include_indices and p in include_indices]
filtered_y_pred = [p for y, p in zip(y_true, y_pred) if y in include_indices and p in include_indices]
new_y_true = [index_map[y] for y in filtered_y_true]
new_y_pred = [index_map[p] for p in filtered_y_pred]
new_class_names = [class_names[i] for i in include_indices]

# Plot
cm = confusion_matrix(new_y_true, new_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=new_class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
# plt.title("Confusion Matrix (Resestor merged into Resistor)")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
