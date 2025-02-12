import os
import numpy as np
import json
import base64
import zlib
import cv2
from sklearn.model_selection import train_test_split

# Define dataset root
DATASET_DIR = '../../../Datasets/PASCAL VOC 2012/'
IMAGE_DIR = os.path.join(DATASET_DIR, 'train/img/')  # Directory containing images
JSON_DIR = os.path.join(DATASET_DIR, 'train/ann/')    # Directory containing JSON files

def decode_bitmap(bitmap_data):
    """Decode base64 bitmap data from JSON"""
    decoded = base64.b64decode(bitmap_data)
    try:
        decompressed = zlib.decompress(decoded)
        return np.frombuffer(decompressed, dtype=np.uint8)
    except:
        return np.frombuffer(decoded, dtype=np.uint8)

# def create_mask_from_json(json_path, image_size=(500, 375)):
#     """Create segmentation mask from JSON file"""
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     # Create a multi-class mask (you can adjust number of classes as needed)
#     mask = np.zeros(image_size, dtype=np.uint8)
    
#     for obj in data['objects']:
#         class_name = obj['classTitle']
#         bitmap_data = obj['bitmap']['data']
        
#         try:
#             mask_data = decode_bitmap(bitmap_data)
#             obj_mask = np.unpackbits(mask_data)[:image_size[0] * image_size[1]]
#             obj_mask = obj_mask.reshape(image_size)
            
#             # You can assign different values for different classes
#             # For now, setting all objects to 1
#             mask = np.logical_or(mask, obj_mask)
#         except Exception as e:
#             print(f"Error processing mask in {json_path}: {e}")
    
#     return mask.astype(np.uint8) * 255

def create_mask_from_json(json_path):
    """Create segmentation mask from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image size from JSON data
    height = data['size']['height']
    width = data['size']['width']
    # Creating empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for obj in data['objects']:
        try:
            # Get bitmap data and origin coordinates
            bitmap_data = obj['bitmap']['data']
            origin = obj['bitmap']['origin']  # [x, y]
            
            # Decode bitmap
            mask_data = decode_bitmap(bitmap_data)
            
            # Calculate actual object dimensions
            total_pixels = len(mask_data) * 8  # Each byte is 8 bits
            
            # Create object mask with correct dimensions
            obj_mask = np.unpackbits(mask_data)[:total_pixels]
            width_obj = origin[0] + int(np.sqrt(total_pixels))
            height_obj = total_pixels // width_obj if width_obj > 0 else 0
            
            try:
                obj_mask = obj_mask.reshape((height_obj, width_obj))
                
                # Place object mask at correct position
                y_start = origin[1]
                x_start = origin[0]
                y_end = min(y_start + height_obj, height)
                x_end = min(x_start + width_obj, width)
                
                obj_height = y_end - y_start
                obj_width = x_end - x_start
                
                if obj_height > 0 and obj_width > 0:
                    mask[y_start:y_end, x_start:x_end] = np.logical_or(
                        mask[y_start:y_end, x_start:x_end],
                        obj_mask[:obj_height, :obj_width]
                    )
            except Exception as e:
                print(f"Skipping object in {json_path} due to reshape error: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing object in {json_path}: {e}")
            continue
    
    return mask.astype(np.uint8) * 255

# def create_mask_from_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     height = data['size']['height']
#     width = data['size']['width']
#     mask = np.zeros((height, width), dtype=np.uint8)
    
#     for obj in data['objects']:
#         try:
#             bitmap_data = obj['bitmap']['data']
#             origin = obj['bitmap']['origin']  # [x, y]
            
#             mask_data = decode_bitmap(bitmap_data)
#             total_pixels = len(mask_data) * 8  # Each byte is 8 bits
            
#             obj_mask = np.unpackbits(mask_data)[:total_pixels]
#             width_obj = int(np.sqrt(total_pixels))
#             height_obj = total_pixels // width_obj if width_obj > 0 else 0
            
#             obj_mask = np.reshape(obj_mask, (height_obj, width_obj))
            
#             y_start = origin[1]
#             x_start = origin[0]
#             y_end = min(y_start + height_obj, height)
#             x_end = min(x_start + width_obj, width)
            
#             obj_height = y_end - y_start
#             obj_width = x_end - x_start
            
#             if obj_height > 0 and obj_width > 0:
#                 mask[y_start:y_end, x_start:x_end] = np.logical_or(
#                     mask[y_start:y_end, x_start:x_end],
#                     obj_mask[:obj_height, :obj_width]
#                 )
#         except Exception as e:
#             print(f"Error processing object in {json_path}: {e}")
    
#     return mask.astype(np.uint8) * 255


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

# def train_generator():
#     for image, mask in data_generator(train_images, train_jsons, IMAGE_DIR, JSON_DIR):
#         image = cv2.resize(image, (512, 512))
#         mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
#         image = np.reshape(image, (512, 512, 3))  # Add this line
#         mask = np.reshape(mask, (512, 512, 1))  # Add this line
#         yield image, mask

# def val_generator():
#     for image, mask in data_generator(val_images, val_jsons, IMAGE_DIR, JSON_DIR):
#         image = cv2.resize(image, (512, 512))
#         mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
#         image = np.reshape(image, (512, 512, 3))  # Add this line
#         mask = np.reshape(mask, (512, 512, 1))  # Add this line
#         yield image, mask

def train_generator(batch_size=1):
    while True:
        batch_images = []
        batch_masks = []
        for image, mask in data_generator(train_images, train_jsons, IMAGE_DIR, JSON_DIR):
            # Resize to 1024x1024 to match model output
            image = cv2.resize(image, (1024, 1024))
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            
            # Ensure mask is integer type for sparse categorical crossentropy
            mask = mask.astype(np.int32)
            
            batch_images.append(image)
            batch_masks.append(mask)
            
            if len(batch_images) == batch_size:
                X = np.array(batch_images)
                y = np.array(batch_masks)
                yield X, y
                batch_images = []
                batch_masks = []

def val_generator(batch_size=1):
    while True:
        batch_images = []
        batch_masks = []
        for image, mask in data_generator(val_images, val_jsons, IMAGE_DIR, JSON_DIR):
            # Resize to 1024x1024 to match model output
            image = cv2.resize(image, (1024, 1024))
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            # Normalize image
            image = image.astype(np.float32) / 255.0
            
            # Ensure mask is integer type for sparse categorical crossentropy
            mask = mask.astype(np.int32)
            
            batch_images.append(image)
            batch_masks.append(mask)
            
            if len(batch_images) == batch_size:
                X = np.array(batch_images)
                y = np.array(batch_masks)
                yield X, y
                batch_images = []
                batch_masks = []
# def train_generator(batch_size=1):
#     while True:  # Make the generator infinite for training
#         batch_images = []
#         batch_masks = []
#         for image, mask in data_generator(train_images, train_jsons, IMAGE_DIR, JSON_DIR):
#             # Preprocess image and mask
#             image = cv2.resize(image, (512, 512))
#             mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
#             # Normalize image
#             image = image.astype(np.float32) / 255.0
            
#             # Reshape and add to batch
#             image = np.reshape(image, (512, 512, 3))
#             mask = np.reshape(mask, (512, 512, 1))
            
#             batch_images.append(image)
#             batch_masks.append(mask)
            
#             if len(batch_images) == batch_size:
#                 # Convert to numpy arrays with explicit shapes
#                 X = np.array(batch_images)
#                 y = np.array(batch_masks)
#                 yield X, y
#                 batch_images = []
#                 batch_masks = []

# def val_generator(batch_size=1):
#     while True:  # Make the generator infinite for validation
#         batch_images = []
#         batch_masks = []
#         for image, mask in data_generator(val_images, val_jsons, IMAGE_DIR, JSON_DIR):
#             # Preprocess image and mask
#             image = cv2.resize(image, (512, 512))
#             mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
#             # Normalize image
#             image = image.astype(np.float32) / 255.0
            
#             # Reshape and add to batch
#             image = np.reshape(image, (512, 512, 3))
#             mask = np.reshape(mask, (512, 512, 1))
            
#             batch_images.append(image)
#             batch_masks.append(mask)
            
#             if len(batch_images) == batch_size:
#                 # Convert to numpy arrays with explicit shapes
#                 X = np.array(batch_images)
#                 y = np.array(batch_masks)
#                 yield X, y
#                 batch_images = []
#                 batch_masks = []

# Example usage:
'''
# Get a batch of data
for image, mask in train_gen:
    # Process your image and mask
    # image shape: (height, width, 3)
    # mask shape: (height, width)
    break  # Remove this if you want to process all images
'''