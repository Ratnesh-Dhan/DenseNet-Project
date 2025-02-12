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
from tqdm import tqdm

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(512, 512)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.images = []
        self.masks = []
        self.class_mapping = {}
        self.next_class_id = 0
        
        json_dir = data_dir + 'ann/'
        image_dir = data_dir + 'img/'
        
        print(f"Loading dataset from {data_dir}")
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        for json_file in tqdm(json_files, desc="Processing annotations"):
            try:
                with open(os.path.join(json_dir, json_file), 'r') as f:
                    data = json.load(f)
                
                # Use int32 instead of uint8 to handle larger class IDs
                mask = np.zeros((data['size']['height'], data['size']['width']), dtype=np.int32)
                
                for obj in data['objects']:
                    try:
                        bitmap_data = base64.b64decode(obj['bitmap']['data'])
                        decompressed_data = zlib.decompress(bitmap_data)
                        
                        bitmap_image = Image.open(io.BytesIO(decompressed_data))
                        bitmap_array = np.array(bitmap_image)
                        
                        origin_x, origin_y = obj['bitmap']['origin']
                        class_id = self._map_class_id(obj['classId'])  # Map the class ID
                        
                        h, w = bitmap_array.shape
                        if origin_y + h <= mask.shape[0] and origin_x + w <= mask.shape[1]:
                            mask[origin_y:origin_y + h, origin_x:origin_x + w] = class_id
                        
                    except Exception as e:
                        print(f"Error processing object in {json_file}: {str(e)}")
                        continue
                
                # Resize mask to target size
                mask_pil = Image.fromarray(mask.astype(np.uint32))
                mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)
                self.masks.append(np.array(mask_pil))
                
                # # Find and resize image
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
        print(f"Found {len(self.class_mapping)} unique classes")

    def __len__(self):
        return len(self.images)
    
    def _map_class_id(self, original_id):
        """Map original class IDs to consecutive integers starting from 0"""
        if original_id not in self.class_mapping:
            self.class_mapping[original_id] = self.next_class_id
            self.next_class_id += 1
        return self.class_mapping[original_id]
    
    # def __getitem__(self, idx):
    #     # Load and resize image
    #     image = Image.open(self.images[idx]).convert('RGB')
    #     image = image.resize(self.target_size, Image.BILINEAR)
    #     mask = self.masks[idx]
        
    #     if self.transform:
    #         image = self.transform(image)
    #         mask = torch.from_numpy(mask).long()  # Convert to long tensor for CrossEntropyLoss
        
    #     return image, mask

    def __getitem__(self, idx):
        # Load and resize image
        image = Image.open(self.images[idx]).convert('RGB')
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = self.masks[idx]
        
        # Resize mask to match the output size
        mask = Image.fromarray(mask.astype(np.uint32))
        mask = mask.resize(self.target_size, Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask
    
# [Rest of the code remains the same: DeepLabV3Plus, ASPP, Decoder classes]

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

# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         return self.relu(x)
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if isinstance(x, dict):
            x = x['features']
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)

# class Decoder(nn.Module):
#     def __init__(self, num_classes):
#         super(Decoder, self).__init__()
#         self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(256)
#         self.conv2 = nn.Conv2d(256, num_classes, 1)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         return x
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x)
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
    batch_size = 4
    num_epochs = 50
    learning_rate = 0.001

    target_size = (512, 512)  # Fixed size for all images
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = '../../Datasets/PASCAL VOC 2012/train/'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    print("Initializing dataset...")
    dataset = SegmentationDataset(dataset_path, transform=transform, target_size=target_size)
    num_classes = len(dataset.class_mapping)

    train_loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DeepLabV3Plus(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    torch.save(model.state_dict(), 'deeplabv3_efficientnet_final.pth')
    print("Training completed!")

if __name__ == '__main__':
    main()