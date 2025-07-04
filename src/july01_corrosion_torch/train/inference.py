import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from load_pretrained_mask_rcnn import get_model_instance_segmentation
import torch
import torchvision.transforms as T

# image_path = r"D:\NML 2nd working directory\corrosion sample piece\dataset\images\IMG_20250425_104935.jpg"
image_path = r"C:\Users\NDT Lab\Pictures\sir send\shortName.jpg"
num_classes = 3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(r"../models/mask_rcnn_july04_best_model_epoch_9.pth"))
model.to(device)
model.eval()

image = Image.open(image_path).convert("RGB")
transform = T.ToTensor()
img_tensor = transform(image).to(device)

# Inference
with torch.no_grad():
    prediction = model([img_tensor])

# Parse Prediction
output = prediction[0]
pred_boxes = output['boxes'].cpu().numpy()
pred_labels = output['labels'].cpu().numpy()
pred_scores = output['scores'].cpu().numpy()
pred_masks = output['masks'].cpu().numpy()

# Filter by confidence score
keep = pred_scores > 0.7
pred_boxes = pred_boxes[keep]
pred_labels = pred_labels[keep]
pred_masks = pred_masks[keep]
pred_scores = pred_scores[keep]

# Visualize masks
label_map = {1: 'corrosion', 2: 'Piece', 3: 'background'}
colors = ['r', 'g', 'b', 'y', 'c']

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for i in range(len(pred_masks)):
    mask = pred_masks[i][0]  # remove 1xHxW to HxW
    color = colors[i % len(colors)]
    ax.imshow(np.ma.masked_where(mask < 0.5, mask), cmap='jet', alpha=0.5)

    xmin, ymin, xmax, ymax = pred_boxes[i]
    label = label_map.get(pred_labels[i], "unknown")
    score = pred_scores[i]

    # Draw bounding box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2,
                             edgecolor='color', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin - 5, f"{label}: {score:.2f}", color='white',
            # bbox=dict(facecolor=color, alpha=0.7), fontsize=10)
            bbox=dict(facecolor='color', alpha=0.7), fontsize=10)

ax.axis("off")
plt.tight_layout()
plt.show()