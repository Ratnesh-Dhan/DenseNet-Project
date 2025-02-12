import torch
from PIL import Image
import torchvision.transforms as transforms

def predict_segmentation(image_path, model_path, num_classes):
    # Load the trained model
    model = DeepLabV3Plus(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().numpy()
    
    return prediction

# Example usage
image_path = 'path/to/your/test/image.jpg'
model_path = 'deeplabv3_efficientnet.pth'
num_classes = 21 # Update based on your classes

# Get prediction
segmentation_map = predict_segmentation(image_path, model_path, num_classes)