import os
import numpy as np
import json
import base64
import zlib
import cv2
from sklearn.model_selection import train_test_split

# Define dataset root
DATASET_DIR = '../../../Datasets/PASCAL VOC 2012/'
IMAGE_DIR = os.path.join(DATASET_DIR, 'train/img/')
JSON_DIR = os.path.join(DATASET_DIR, 'train/ann/')

def decode_bitmap(bitmap_data):
    """Decode base64 bitmap data from JSON"""
    decoded = base64.b64decode(bitmap_data)
    try:
        decompressed = zlib.decompress(decoded)
        return np.frombuffer(decompressed, dtype=np.uint8)
    except:
        return np.frombuffer(decoded, dtype=np.uint8)

def create_mask_from_json(json_path):
    """Create segmentation mask from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get original image size from JSON
    original_size = (data['size']['height'], data['size']['width'])
    mask = np.zeros(original_size, dtype=np.uint8)
    
    for obj in data['objects']:
        bitmap_data = obj['bitmap']['data']
        origin = obj['bitmap']['origin']
        
        try:
            mask_data = decode_bitmap(bitmap_data)
            # Calculate the actual size of this object's mask
            mask_size = len(mask_data) * 8  # 8 bits per byte
            obj_mask = np.unpackbits(mask_data)[:mask_size]
            
            # Calculate dimensions for this specific object
            width = int(np.sqrt(mask_size))
            height = mask_size // width
            obj_mask = obj_mask.reshape((height, width))
            
            # Place the object mask at its origin
            y, x = origin[1], origin[0]
            h, w = obj_mask.shape
            mask[y:y+h, x:x+w] = np.logical_or(mask[y:y+h, x:x+w], obj_mask)
            
        except Exception as e:
            print(f"Error processing object in {json_path}: {e}")
    
    return mask.astype(np.uint8) * 255

# Load image and JSON filenames
images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])
jsons = sorted([f for f in os.listdir(JSON_DIR) if f.endswith('.json')])

# Ensure image-JSON pairs are correctly matched
assert len(images) == len(jsons), "Mismatch in images and JSON files"

# Split into train and validation sets
train_images, val_images, train_jsons, val_jsons = train_test_split(
    images, jsons, test_size=0.2, random_state=42
)

def data_generator(image_list, json_list, image_dir, json_dir):
    """Yields image-mask pairs for training/validation"""
    for img_name, json_name in zip(image_list, json_list):
        img_path = os.path.join(image_dir, img_name)
        json_path = os.path.join(json_dir, json_name)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mask from JSON
        mask = create_mask_from_json(json_path)

        yield image, mask

def train_generator():
    for image, mask in data_generator(train_images, train_jsons, IMAGE_DIR, JSON_DIR):
        image = cv2.resize(image, (512, 512))
        image = image / 255.0  # Normalize image
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        yield image, mask

def val_generator():
    for image, mask in data_generator(val_images, val_jsons, IMAGE_DIR, JSON_DIR):
        image = cv2.resize(image, (512, 512))
        image = image / 255.0  # Normalize image
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        yield image, mask

# Creating generators for training and validation
train_gen = train_generator()
val_gen = val_generator()