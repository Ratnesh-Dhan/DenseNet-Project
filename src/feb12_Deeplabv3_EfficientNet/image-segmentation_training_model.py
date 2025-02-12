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
import zlib
from tqdm import tqdm  # for progress bars

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.masks = []
        
        json_dir = data_dir + 'ann/'
        image_dir = data_dir + 'img/'
        
        print(f"Loading dataset from {data_dir}")
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        for json_file in tqdm(json_files, desc="Processing annotations"):
            try:
                with open(os.path.join(json_dir, json_file), 'r') as f:
                    data = json.load(f)
                
                mask = np.zeros((data['size']['height'], data['size']['width']), dtype=np.uint8)
                
                for obj in data['objects']:
                    try:
                        # Decode and decompress bitmap data
                        bitmap_data = base64.b64decode(obj['bitmap']['data'])
                        decompressed_data = zlib.decompress(bitmap_data)
                        
                        bitmap_image = Image.open(io.BytesIO(decompressed_data))
                        bitmap_array = np.array(bitmap_image)
                        
                        origin_x, origin_y = obj['bitmap']['origin']
                        class_id = obj['classId']
                        
                        # Ensure we don't exceed mask boundaries
                        h, w = bitmap_array.shape
                        if origin_y + h <= mask.shape[0] and origin_x + w <= mask.shape[1]:
                            mask[origin_y:origin_y + h, origin_x:origin_x + w] = class_id
                        
                    except Exception as e:
                        print(f"Error processing object in {json_file}: {str(e)}")
                        continue
                
                self.masks.append(mask)
                
                # # Check for image file with different possible extensions
                # for ext in ['.jpg', '.png', '.jpeg']:
                #     image_file = json_file.replace('.json', ext)
                #     image_path = os.path.join(image_dir, image_file)
                #     if os.path.exists(image_path):
                #         self.images.append(image_path)
                #         break

                image_file = json_file.strip('.json')
                image_path = os.path.join(image_dir, image_file)
                if os.path.exists(image_path):
                    self.images.append(image_path)
                        
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                continue
                
        print(f"Loaded {len(self.images)} images and masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = self.masks[idx]
        
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()
        
        return image, mask

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 
                                     'nvidia_efficientnet_b0', 
                                     pretrained=True)
        self.aspp = ASPP(1280, 256)
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

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Resize outputs if necessary
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=masks.shape[1:],
                    mode='bilinear',
                    align_corners=False
                )
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss/len(train_loader)})
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

def main():
    # Hyperparameters
    batch_size = 4  # Reduced batch size to avoid memory issues
    num_epochs = 50
    learning_rate = 0.001
    num_classes = 21  # PASCAL VOC has 21 classes
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Set your dataset path here
    dataset_path = '../../Datasets/PASCAL VOC 2012/train/'
    
    print("Initializing dataset...")
    dataset = SegmentationDataset(dataset_path, transform=transform)
    train_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=2)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DeepLabV3Plus(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # Save the final model
    torch.save(model.state_dict(), 'deeplabv3_efficientnet_final.pth')
    print("Training completed!")

if __name__ == '__main__':
    main()