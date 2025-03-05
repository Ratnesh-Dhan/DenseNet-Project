import os
import json, cv2
import base64
import zlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import io, matplotlib.pyplot as plt

# Paths to dataset
TRAIN_IMG_DIR = "../../Datasets/PASCAL VOC 2012/train/img"
TRAIN_ANN_DIR = "../../Datasets/PASCAL VOC 2012/train/ann"
VAL_IMG_DIR = "../../Datasets/PASCAL VOC 2012/val/img"
VAL_ANN_DIR = "../../Datasets/PASCAL VOC 2012/val/ann"

IMG_SIZE = (256, 256)  # Resize images to this size 

def load_image_and_mask(image_path, annotation_path):
    # Load image at original size first to get dimensions
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size
    
    # Create mask at original image dimensions
    original_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    
    # Load annotation JSON
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Process each object's bitmap at original scale
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
            
            # Check if mask would be placed within image boundaries
            x1, y1 = origin[0], origin[1]
            x2, y2 = x1 + w, y1 + h
            
            # Only consider the portion that fits within the image
            if x1 < original_width and y1 < original_height:
                # Calculate valid mask region
                valid_x2 = min(x2, original_width)
                valid_y2 = min(y2, original_height)
                valid_w = valid_x2 - max(0, x1)
                valid_h = valid_y2 - max(0, y1)
                
                if valid_w > 0 and valid_h > 0:
                    # Calculate image and mask slice coordinates
                    img_x1, img_y1 = max(0, x1), max(0, y1)
                    mask_x1 = 0 if x1 >= 0 else -x1
                    mask_y1 = 0 if y1 >= 0 else -y1
                    
                    # Place the mask data into the original sized mask
                    try:
                        original_mask[img_y1:valid_y2, img_x1:valid_x2] = np.maximum(
                            original_mask[img_y1:valid_y2, img_x1:valid_x2],
                            mask_data[mask_y1:mask_y1+valid_h, mask_x1:mask_x1+valid_w]
                        )
                    except ValueError as e:
                        print(f"Error placing mask: {e}, shape: {mask_data.shape}, region: [{mask_y1}:{mask_y1+valid_h}, {mask_x1}:{mask_x1+valid_w}]")
                        continue
    
    # Now resize both image and mask to target size
    resized_image = np.array(original_image.resize(IMG_SIZE)) / 255.0
    resized_mask = cv2.resize(original_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)q
    # Return the resized image and mask
    return resized_image, np.expand_dims(resized_mask, axis=-1)
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

# Model using EfficientNet as backbone with corrected output dimensions
def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    
    # Use EfficientNetB0 as the encoder (backbone)
    base_model = keras.applications.EfficientNetB0(
        include_top=False, 
        input_tensor=inputs,
        weights='imagenet'
    )
    
    # Get the output from the backbone
    # EfficientNetB0 has a reduction factor of 32, so 256x256 becomes 8x8
    encoder_output = base_model.output  # Shape: (None, 8, 8, 1280)
    
    # Decoder pathway with skip connections to ensure correct upsampling
    # Upsampling 1: 8x8 -> 16x16
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(encoder_output)
    x = layers.UpSampling2D((2, 2))(x)  # Now 16x16
    
    # Upsampling 2: 16x16 -> 32x32
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 32x32
    
    # Upsampling 3: 32x32 -> 64x64
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 64x64
    
    # Upsampling 4: 64x64 -> 128x128
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 128x128
    
    # Upsampling 5: 128x128 -> 256x256
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 256x256
    
    # Final layer to produce the segmentation mask
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)  # Single-channel mask at 256x256
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    return model

# Prepare dataset
train_generator = SegmentationDataGenerator(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
val_generator = SegmentationDataGenerator(VAL_IMG_DIR, VAL_ANN_DIR)

# Build and train model
model = build_model()
model.fit(train_generator, validation_data=val_generator, epochs=100)

# Save model
model.save("segmentation_model.h5")


# https://medium.com/@alfred.weirich/transfer-learning-with-keras-tensorflow-an-introduction-51d2766c30ca
# https://www.youtube.com/watch?v=8cN0PiZQl18
# link to study transfer learning