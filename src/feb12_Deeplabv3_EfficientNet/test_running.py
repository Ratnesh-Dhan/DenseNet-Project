import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from DeepLabV3Plus import DeepLabV3Plus



# Load the trained model
model = DeepLabV3Plus(num_classes).to(device)
model.load_state_dict(torch.load('deeplabv3_efficientnet_final.pth'))
model.eval()

# Prepare the input image
image = Image.open('path/to/your/image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0).to(device)

# Run the model and get the output
with torch.no_grad():
    output = model(input_image)

# Postprocess the output
output = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# Visualize the segmentation mask
segmented_image = Image.fromarray(output.astype(np.uint8))
segmented_image.show()
