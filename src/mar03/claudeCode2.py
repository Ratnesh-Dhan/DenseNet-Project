import os
import json, cv2
import base64
import zlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from PIL import Image
import io, matplotlib.pyplot as plt

# Paths to dataset
TRAIN_IMG_DIR = "../../Datasets/PASCAL VOC 2012/train/img"
TRAIN_ANN_DIR = "../../Datasets/PASCAL VOC 2012/train/ann"
VAL_IMG_DIR = "../../Datasets/PASCAL VOC 2012/val/img"
VAL_ANN_DIR = "../../Datasets/PASCAL VOC 2012/val/ann"

IMG_SIZE = (256, 256)  # Resize images to this size 


# 🔹 Dice Loss Function
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


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
    resized_mask = cv2.resize(original_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # Return the resized image and mask
    return resized_image, np.expand_dims(resized_mask, axis=-1)
# Data generator class
class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, img_dir, ann_dir, batch_size=16):
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
def conv_block(x, filters):
    """Convolutional Block with BatchNorm and ReLU"""
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def upsample_block(x, skip, filters):
    """Upsampling Block with Skip Connection"""
    x = layers.Conv2DTranspose(filters, (3, 3), strides=2, padding="same")(x)  # Upsample
    x = layers.Concatenate()([x, skip])  # Add skip connection
    x = conv_block(x, filters)
    return x

def build_model():
    inputs = keras.Input(shape=(256, 256, 3))
    
    # 🔹 Use EfficientNetB3 instead of B0
    base_model = keras.applications.EfficientNetB3(include_top=False, input_tensor=inputs, weights="imagenet")
    
    # Extract encoder features for skip connections
    skips = [
        base_model.get_layer("block2a_expand_activation").output,  # 128x128
        base_model.get_layer("block3a_expand_activation").output,  # 64x64
        base_model.get_layer("block4a_expand_activation").output,  # 32x32
        base_model.get_layer("block6a_expand_activation").output,  # 16x16
    ]
    
    encoder_output = base_model.output  # 8x8 feature map
    
    # 🔹 Decoder with skip connections
    x = upsample_block(encoder_output, skips[3], 512)  # 8x8 → 16x16
    x = upsample_block(x, skips[2], 256)  # 16x16 → 32x32
    x = upsample_block(x, skips[1], 128)  # 32x32 → 64x64
    x = upsample_block(x, skips[0], 64)   # 64x64 → 128x128
    
    # Final upsampling (128x128 → 256x256)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)
    x = conv_block(x, 32)
    
    # 🔹 Final segmentation mask output
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)  # Binary segmentation mask
    
    # Compile model
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_loss,
        metrics=["accuracy"]
    )
    
    return model

# Prepare dataset
train_generator = SegmentationDataGenerator(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
val_generator = SegmentationDataGenerator(VAL_IMG_DIR, VAL_ANN_DIR)

# Build and train model
model = build_model()
history = model.fit(train_generator, validation_data=val_generator, epochs=100)
# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()


# Save model
model.save("mar_4_segmentation_model.h5")
