import os
import json, cv2
import base64
import zlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import io

# Paths to dataset
TRAIN_IMG_DIR = "../../Datasets/PASCAL VOC 2012/train/img"
TRAIN_ANN_DIR = "../../Datasets/PASCAL VOC 2012/train/ann"
VAL_IMG_DIR = "../../Datasets/PASCAL VOC 2012/val/img"
VAL_ANN_DIR = "../../Datasets/PASCAL VOC 2012/val/ann"

IMG_SIZE = (256, 256)  # Resize images to this size

# Function to load and preprocess image and mask
# def load_image_and_mask(image_path, annotation_path):
#     # Load image
#     image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
#     image = np.array(image) / 255.0  # Normalize
    
#     # Load annotation JSON
#     with open(annotation_path, 'r') as f:
#         data = json.load(f)
    
#     # Initialize empty mask
#     mask = np.zeros(IMG_SIZE, dtype=np.uint8)
    
#     for obj in data['objects']:
#         if obj['geometryType'] == 'bitmap':
#             bitmap_data = base64.b64decode(obj['bitmap']['data'])
#             try:
#                 decompressed_data = zlib.decompress(bitmap_data)
#                 bitmap_image = Image.open(io.BytesIO(decompressed_data)).convert('L')
#                 mask_data = np.array(bitmap_image)
#             except Exception as e:
#                 print(f"Error processing bitmap: {e}")
#                 continue
            
#             # Get the origin of the mask
#             origin = obj['bitmap']['origin']
            
#             # Resize mask to match image size
#             mask_resized = np.zeros(IMG_SIZE, dtype=np.uint8)
#             h, w = mask_data.shape
#             # Ensure mask_data matches the target slice size
#             mask_data = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
#             mask_resized[origin[1]:origin[1]+h, origin[0]:origin[0]+w] = mask_data
#             mask = np.maximum(mask, mask_resized)  # Merge masks
    
#     return image, np.expand_dims(mask, axis=-1)  # Add channel dimension
 

def load_image_and_mask(image_path, annotation_path):
    # Load image
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalize

    # Load annotation JSON
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Initialize empty mask
    mask = np.zeros(IMG_SIZE, dtype=np.uint8)

    for obj in data['objects']:
        if obj['geometryType'] == 'bitmap':
            bitmap_data = base64.b64decode(obj['bitmap']['data'])
            try:
                decompressed_data = zlib.decompress(bitmap_data)
                bitmap_image = Image.open(io.BytesIO(decompressed_data)).convert('L')
                mask_data = np.array(bitmap_image)
            except Exception as e:
                print(f"Error processing bitmap: {e}")
                continue

            # Get the origin of the mask
            origin = obj['bitmap']['origin']
            h, w = mask_data.shape

            # Ensure mask fits within IMG_SIZE boundaries
            x1, y1 = origin[0], origin[1]
            x2, y2 = min(x1 + w, IMG_SIZE[0]), min(y1 + h, IMG_SIZE[1])

            # Resize mask to fit the target region
            target_w, target_h = x2 - x1, y2 - y1
            mask_data = cv2.resize(mask_data, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            # Ensure valid assignment by checking shapes
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], mask_data)

    return image, np.expand_dims(mask, axis=-1)  # Add channel dimension
# Data generator class
class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, img_dir, ann_dir, batch_size=8):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.batch_size = batch_size
        self.image_filenames = os.listdir(img_dir)

    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_images = []
        batch_masks = []
        for i in range(self.batch_size):
            img_name = self.image_filenames[idx * self.batch_size + i]
            img_path = os.path.join(self.img_dir, img_name)
            ann_path = os.path.join(self.ann_dir, img_name + ".json")
            if os.path.exists(ann_path):
                image, mask = load_image_and_mask(img_path, ann_path)
                batch_images.append(image)
                batch_masks.append(mask)
        return np.array(batch_images), np.array(batch_masks)

# Model using EfficientNet as backbone
def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    base_model = keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(base_model.output)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)  # Single-channel mask
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Prepare dataset
train_generator = SegmentationDataGenerator(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
val_generator = SegmentationDataGenerator(VAL_IMG_DIR, VAL_ANN_DIR)

# Build and train model
model = build_model()
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save model
model.save("segmentation_model.h5")
