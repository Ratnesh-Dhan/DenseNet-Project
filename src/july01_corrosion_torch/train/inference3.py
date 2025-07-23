import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from load_pretrained_mask_rcnn import get_model_instance_segmentation

# Paths
image_path = r"D:\NML 2nd working directory\Dr. sarma paswan-05-06-2025\Modified\CSS\CSS_OR_3390.jpg"
model_path = r"../working-model/mask_rcnn_july11_with_corrosion_best_model_epoch_11.pth"

# Setup
num_classes = 3
device = torch.device('cpu')
label_map = {1: 'corrosion', 2: 'Piece', 3: 'background'}
label_color_map = {
    1: (1, 0, 0),    # corrosion = red
    2: (0, 1, 0),    # piece = green
    3: (0.5, 0.5, 0.5)  # background (not used here)
}

# Load model
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
transform = T.ToTensor()
img_tensor = transform(image).to(device)

# Inference
with torch.no_grad():
    prediction = model([img_tensor])

# Parse outputs
output = prediction[0]
pred_boxes = output['boxes'].cpu().numpy()
pred_labels = output['labels'].cpu().numpy()
pred_scores = output['scores'].cpu().numpy()
pred_masks = output['masks'].cpu().numpy()

# Filter by confidence
keep = pred_scores > 0.7
pred_labels = pred_labels[keep]
pred_masks = pred_masks[keep]

# Merge masks by class
merged_masks = {}
for label in np.unique(pred_labels):
    masks_for_label = pred_masks[pred_labels == label]
    combined_mask = np.any(masks_for_label > 0.5, axis=0)[0]  # shape: HxW
    merged_masks[label] = combined_mask

# Create output image
image_np = np.array(image) / 255.0
output_img = image_np.copy()

for label, mask in merged_masks.items():
    color = label_color_map.get(label, (0, 0, 1))  # default blue if unknown
    for c in range(3):
        output_img[:, :, c] = np.where(mask,
                                       output_img[:, :, c] * 0.5 + color[c] * 0.5,
                                       output_img[:, :, c])

# Plot original and masked output
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis("off")

ax2.imshow(output_img)
ax2.set_title("Stitched Masks by Class")
ax2.axis("off")

plt.tight_layout()
plt.show()
