import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from load_pretrained_mask_rcnn import get_model_instance_segmentation

# Set path to image and model
# image_path = r"C:\Users\NDT Lab\Pictures\sir send\shortName.jpg"
image_path = r"D:\NML 2nd working directory\Dr. sarma paswan-05-06-2025\Modified\CSS\CSS_SY_3391.jpg"
model_path = r"../working-model/mask_rcnn_july11_with_corrosion_best_model_epoch_11.pth"

# Set number of classes
num_classes = 3

# Use CPU for inference
device = torch.device('cpu')

# Load model
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
transform = T.ToTensor()
img_tensor = transform(image).to(device)

# Inference
with torch.no_grad():
    prediction = model([img_tensor])

# Parse output
output = prediction[0]
pred_boxes = output['boxes'].cpu().numpy()
pred_labels = output['labels'].cpu().numpy()
pred_scores = output['scores'].cpu().numpy()
pred_masks = output['masks'].cpu().numpy()

# Filter predictions by confidence score
keep = pred_scores > 0.7
pred_boxes = pred_boxes[keep]
pred_labels = pred_labels[keep]
pred_masks = pred_masks[keep]
pred_scores = pred_scores[keep]

# Label and color mappings
label_map = {1: 'corrosion', 2: 'Piece', 3: 'background'}
label_color_map = {1: (1, 0, 0),   # red for corrosion
                   2: (0, 1, 0),   # green for piece
                   3: (0.5, 0.5, 0.5)}  # gray for background

# Create output mask overlay image
image_np = np.array(image) / 255.0  # Normalize to [0, 1]
output_img = image_np.copy()

for i in range(len(pred_masks)):
    mask = pred_masks[i][0] > 0.5
    label_id = pred_labels[i]
    color = label_color_map.get(label_id, (0, 0, 1))  # fallback: blue

    for c in range(3):
        output_img[:, :, c] = np.where(mask, 
                                       output_img[:, :, c] * 0.5 + color[c] * 0.5,
                                       output_img[:, :, c])

# Plot original and output side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Original
ax1.imshow(image)
ax1.set_title("Input Image")
ax1.axis("off")

# Masked Output
ax2.imshow(output_img)
ax2.set_title("Output with Masks")
ax2.axis("off")

plt.tight_layout()
plt.show()
