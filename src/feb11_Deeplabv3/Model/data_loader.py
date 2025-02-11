import os
import numpy as np
import json
import base64
import zlib
import cv2
from sklearn.model_selection import train_test_split

# Define dataset root
DATASET_DIR = '../../../Datasets/PASCAL_VOC_2012'
IMAGE_DIR = os.path.join(DATASET_DIR, 'train/img')  # Directory containing images
JSON_DIR = os.path.join(DATASET_DIR, 'train/ann')    # Directory containing JSON files

def decode_bitmap(bitmap_data):
    """Decode base64 bitmap data from JSON"""
    decoded = base64.b64decode(bitmap_data)
    try:
        decompressed = zlib.decompress(decoded)
        return np.frombuffer(decompressed, dtype=np.uint8)
    except:
        return np.frombuffer(decoded, dtype=np.uint8)

def create_mask_from_json(json_path, image_size=(500, 375)):
    """Create segmentation mask from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a multi-class mask (you can adjust number of classes as needed)
    mask = np.zeros(image_size, dtype=np.uint8)
    
    for obj in data['objects']:
        class_name = obj['classTitle']
        bitmap_data = obj['bitmap']['data']
        
        try:
            mask_data = decode_bitmap(bitmap_data)
            obj_mask = np.unpackbits(mask_data)[:image_size[0] * image_size[1]]
            obj_mask = obj_mask.reshape(image_size)
            
            # You can assign different values for different classes
            # For now, setting all objects to 1
            mask = np.logical_or(mask, obj_mask)
        except Exception as e:
            print(f"Error processing mask in {json_path}: {e}")
    
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

# Creating generators for training and validation
train_gen = data_generator(train_images, train_jsons, IMAGE_DIR, JSON_DIR)
val_gen = data_generator(val_images, val_jsons, IMAGE_DIR, JSON_DIR)

# Example usage:
'''
# Get a batch of data
for image, mask in train_gen:
    # Process your image and mask
    # image shape: (height, width, 3)
    # mask shape: (height, width)
    break  # Remove this if you want to process all images
'''