#IN THIS WE ARE GOING TO USE RESNET101 OF DEEPLABV3PLUS

import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import cv2
model_name = "best_deeplabv3plus_resnet101_cityscapes_os16.pth"