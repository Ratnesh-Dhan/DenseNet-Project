import torch
import torch.nn as nn

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
        if isinstance(x, dict):
            x = x['features']
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
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x
