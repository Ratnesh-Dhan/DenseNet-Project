#IN THIS WE ARE GOING TO USE RESNET101 OF DEEPLABV3PLUS

import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import cv2
from matplotlib import pyplot as plt
import numpy as np

model_name = "../../TrainedModel/best_deeplabv3plus_resnet101_cityscapes_os16.pth"
model = deeplabv3_resnet101(pretrained=True) # Set pretrained=True if you want to use the pretrained weights from torchvision
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "../img/scenery.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():  # Disable gradient calculation
    output = model(input_batch)['out']  # Get the output from the model

# Get the predicted class for each pixel
output_predictions = torch.argmax(output, dim=1)  # Shape: (1, num_classes, height, width)
mask = output_predictions.squeeze().cpu().numpy()  # Remove batch dimension and convert to numpy array

# Visualize the original image and the segmentation mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap='jet', alpha=0.5)  # Use a colormap to visualize the mask
plt.axis("off")

plt.show()
