import os
import torch
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET
from torchvision.transforms import functional as F


# ====================== XML DATASET (Pascal VOC Format) ======================
class XMLDataset(Dataset):
    def __init__(self, split, root_dir, classes_file):
        """
        Args:
            split: 'train', 'val', or 'test'
            root_dir: Root directory containing images/ and annotations/
            classes_file: Path to text file with class names (one per line)
        """
        self.img_dir = os.path.join(root_dir, "images", split)
        self.ann_dir = os.path.join(root_dir, "annotations", split)
        
        # Load class names
        with open(classes_file) as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Create class name to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all images
        self.imgs = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        print(f"Loaded {len(self.imgs)} images from {split} split")

    def __len__(self):
        return len(self.imgs)

    def parse_xml(self, xml_path):
        """Parse Pascal VOC XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        # Get image size (optional, for validation)
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            width = height = None
        
        # Parse all objects
        for obj in root.findall('object'):
            # Get class name
            name = obj.find('name').text
            
            # Skip if class not in our list
            if name not in self.class_to_idx:
                print(f"Warning: Unknown class '{name}' in {xml_path}")
                continue
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Validate box
            if xmax <= xmin or ymax <= ymin:
                print(f"Warning: Invalid box in {xml_path}: [{xmin}, {ymin}, {xmax}, {ymax}]")
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name] + 1)  # +1 for background
        
        return boxes, labels, (width, height)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # XML file has same name as image but .xml extension
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(self.ann_dir, xml_name)
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Parse XML annotation
        boxes, labels, (xml_w, xml_h) = [], [], (None, None)
        
        if os.path.exists(xml_path):
            boxes, labels, (xml_w, xml_h) = self.parse_xml(xml_path)
            
            # Validate image size matches XML (if available)
            if xml_w is not None and (xml_w != w or xml_h != h):
                print(f"Warning: Image size mismatch in {img_name}: "
                      f"Image={w}x{h}, XML={xml_w}x{xml_h}")
        else:
            print(f"Warning: No XML file found for {img_name}")
        
        # Convert to tensor
        img_tensor = F.to_tensor(img)
        
        # Resize to 300x300 (SSD300 requirement)
        img_tensor = F.resize(img_tensor, [300, 300])
        
        # Scale boxes to new dimensions
        scale_x = 300 / w
        scale_y = 300 / h
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
            boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Normalize with ImageNet stats
        img_tensor = F.normalize(img_tensor, 
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        
        target = {
            "boxes": boxes,
            "labels": labels,
        }
        
        return img_tensor, target
