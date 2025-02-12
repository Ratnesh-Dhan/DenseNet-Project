#This is Deeplabv3 with backbone of EfficientNet for image segmentatin with dataset Pascal Voc 2012
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import base64
from PIL import Image
import io


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.masks = []
        
        json_dir = data_dir + 'ann/'
        image_dir = data_dir + 'img/'
        
        # Check if directories exist
        if not os.path.exists(json_dir):
            raise ValueError(f"JSON directory not found: {json_dir}")
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
            
        # Get all JSON files
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(os.path.join(json_dir, json_file), 'r') as f:
                    data = json.load(f)
                
                # Create mask from objects
                height = data['size']['height']
                width = data['size']['width']
                mask = np.zeros((height, width), dtype=np.uint8)
                
                for obj in data['objects']:
                    try:
                        # Print the bitmap data length for debugging
                        print(f"Processing object in {json_file}, bitmap data length: {len(obj['bitmap']['data'])}")
                        
                        # Try to decode bitmap data
                        bitmap_data = base64.b64decode(obj['bitmap']['data'])
                        
                        try:
                            bitmap_image = Image.open(io.BytesIO(bitmap_data))
                            bitmap_array = np.array(bitmap_image)
                            
                            origin_x, origin_y = obj['bitmap']['origin']
                            class_id = obj['classId']
                            
                            # Check array dimensions before assignment
                            if origin_y + bitmap_array.shape[0] <= height and origin_x + bitmap_array.shape[1] <= width:
                                mask[origin_y:origin_y + bitmap_array.shape[0],
                                     origin_x:origin_x + bitmap_array.shape[1]] = class_id
                            else:
                                print(f"Warning: bitmap dimensions exceed mask size in {json_file}")
                                
                        except Exception as e:
                            print(f"Error processing bitmap in {json_file}: {str(e)}")
                            continue
                            
                    except KeyError as e:
                        print(f"Missing key in object data in {json_file}: {str(e)}")
                        continue
                
                self.masks.append(mask)
                
                # Get corresponding image file
                image_file = json_file.replace('.json', '.jpg')  # Try .jpg extension
                image_path = os.path.join(image_dir, image_file)
                
                # Check if image exists with different extensions if .jpg not found
                if not os.path.exists(image_path):
                    for ext in ['.png', '.jpeg', '.JPEG', '.PNG']:
                        alt_image_path = os.path.join(image_dir, json_file.replace('.json', ext))
                        if os.path.exists(alt_image_path):
                            image_path = alt_image_path
                            break
                
                if not os.path.exists(image_path):
                    print(f"Image not found for {json_file}")
                    continue
                    
                self.images.append(image_path)
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            mask = self.masks[idx]
            
            if self.transform:
                image = self.transform(image)
                mask = torch.from_numpy(mask)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            raise e
        
# DeepLabv3+ with EfficientNet backbone
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.aspp = ASPP(1280, 256)  # 1280 is the output channels of EfficientNet-B0
        self.decoder = Decoder(num_classes)
        
    def forward(self, x):
        features = self.backbone.extract_features(x)
        aspp_features = self.aspp(features)
        output = self.decoder(aspp_features)
        return output

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Add other ASPP components here
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Main training script
def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 21  # Update based on your dataset
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SegmentationDataset('../../Datasets/PASCAL VOC 2012/train/', transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3Plus(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # Save the model
    torch.save(model.state_dict(), 'deeplabv3_efficientnet.pth')

if __name__ == '__main__':
    main()
