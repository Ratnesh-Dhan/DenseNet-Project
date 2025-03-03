import os
import json
import base64
import zlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt

# Paths to dataset
TRAIN_IMG_DIR = "../../Datasets/PASCAL VOC 2012/train/img"
TRAIN_ANN_DIR = "../../Datasets/PASCAL VOC 2012/train/ann"
VAL_IMG_DIR = "../../Datasets/PASCAL VOC 2012/val/img"
VAL_ANN_DIR = "../../Datasets/PASCAL VOC 2012/val/ann"

IMG_SIZE = (256, 256)  # Resize images to this size

# Improved function to load and preprocess image and mask
def load_image_and_mask(image_path, annotation_path):
    # Load image at original size first
    original_image = Image.open(image_path).convert("RGB")
    original_width, original_height = original_image.size
    
    # Create mask at original image dimensions
    original_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    
    # Load annotation JSON
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading annotation {annotation_path}: {e}")
        # Return empty mask if annotation can't be loaded
        resized_image = np.array(original_image.resize(IMG_SIZE)) / 255.0
        resized_mask = np.zeros(IMG_SIZE, dtype=np.uint8)
        return resized_image, np.expand_dims(resized_mask, axis=-1)

    # Process each object's bitmap
    for obj in data.get('objects', []):
        if obj.get('geometryType') == 'bitmap' and 'bitmap' in obj:
            try:
                bitmap_data = base64.b64decode(obj['bitmap']['data'])
                decompressed_data = zlib.decompress(bitmap_data)
                bitmap_image = Image.open(io.BytesIO(decompressed_data)).convert('L')
                mask_data = np.array(bitmap_image)
                
                # Get the origin of the mask
                origin = obj['bitmap']['origin']
                h, w = mask_data.shape
                
                # Place mask on original canvas
                x1, y1 = origin[0], origin[1]
                
                # Skip if origin is completely outside the image
                if x1 >= original_width or y1 >= original_height:
                    continue
                    
                # Calculate valid regions
                x2 = min(x1 + w, original_width)
                y2 = min(y1 + h, original_height)
                
                # Skip if dimensions are invalid
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Calculate valid source region in mask_data
                src_x1 = max(0, -x1)
                src_y1 = max(0, -y1)
                src_x2 = src_x1 + (x2 - max(0, x1))
                src_y2 = src_y1 + (y2 - max(0, y1))
                
                # Place mask data
                try:
                    original_mask[max(0, y1):y2, max(0, x1):x2] = np.maximum(
                        original_mask[max(0, y1):y2, max(0, x1):x2],
                        mask_data[src_y1:src_y2, src_x1:src_x2]
                    )
                except Exception as e:
                    print(f"Error placing mask: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error processing bitmap: {e}")
                continue
    
    # Now resize both image and mask to target size
    resized_image = np.array(original_image.resize(IMG_SIZE)) / 255.0
    resized_mask = cv2.resize(original_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # Apply threshold to ensure binary mask (important!)
    binary_mask = (resized_mask > 0).astype(np.uint8)
    
    # Add debug visualization
    # This will help diagnose if masks are being loaded correctly
    if np.max(binary_mask) > 0:  # Only save non-empty masks
        debug_dir = "debug_masks"
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(resized_image)
        plt.title("Image")
        plt.subplot(132)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Mask")
        plt.subplot(133)
        overlay = resized_image.copy()
        overlay[..., 0] = np.where(binary_mask > 0, 1.0, overlay[..., 0])
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.savefig(os.path.join(debug_dir, f"debug_{base_name}.png"))
        plt.close()
    
    return resized_image, np.expand_dims(binary_mask, axis=-1)

# Data generator with data augmentation
class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, img_dir, ann_dir, batch_size=8, is_training=True):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.batch_size = batch_size
        self.is_training = is_training
        self.image_filenames = [f for f in os.listdir(img_dir) 
                               if os.path.exists(os.path.join(ann_dir, f + ".json"))]
        # Print dataset size
        print(f"Found {len(self.image_filenames)} valid image-annotation pairs")

    def __len__(self):
        return max(1, len(self.image_filenames) // self.batch_size)

    def __getitem__(self, idx):
        batch_images = []
        batch_masks = []
        
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_filenames))
        
        for i in range(start_idx, end_idx):
            img_name = self.image_filenames[i]
            img_path = os.path.join(self.img_dir, img_name)
            ann_path = os.path.join(self.ann_dir, img_name + ".json")
            
            try:
                image, mask = load_image_and_mask(img_path, ann_path)
                
                # Skip images with empty masks during training
                if self.is_training and np.max(mask) == 0:
                    continue
                    
                # Simple data augmentation for training
                if self.is_training and np.random.rand() > 0.5:
                    # Horizontal flip
                    image = np.fliplr(image)
                    mask = np.fliplr(mask)
                
                batch_images.append(image)
                batch_masks.append(mask)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
        # Ensure we have at least one sample
        if len(batch_images) == 0:
            # Create a dummy sample to avoid empty batch error
            dummy_img = np.zeros((256, 256, 3), dtype=np.float32)
            dummy_mask = np.zeros((256, 256, 1), dtype=np.uint8)
            batch_images.append(dummy_img)
            batch_masks.append(dummy_mask)
            
        return np.array(batch_images), np.array(batch_masks)

# Custom loss function: Combination of Binary Cross-Entropy and Dice Loss
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# UNet model with integrated EfficientNet features
def build_unet_model():
    inputs = keras.Input(shape=(256, 256, 3))
    
    # Encoder pathway
    # Using smaller model to avoid overfitting
    base_model = keras.applications.EfficientNetB0(
        include_top=False, 
        input_tensor=inputs,
        weights='imagenet'
    )
    
    # Get features from different levels for skip connections
    # These layer names are specific to EfficientNetB0
    skips = [
        base_model.get_layer('block2a_expand_activation').output,  # 64x64
        base_model.get_layer('block3a_expand_activation').output,  # 32x32
        base_model.get_layer('block4a_expand_activation').output,  # 16x16
        base_model.get_layer('block6a_expand_activation').output,  # 8x8
    ]
    
    x = base_model.output  # 8x8
    
    # Decoder pathway with skip connections
    # Upsampling 1: 8x8 -> 16x16
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 16x16
    x = layers.Concatenate()([x, skips[3]])
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    
    # Upsampling 2: 16x16 -> 32x32
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 32x32
    x = layers.Concatenate()([x, skips[2]])
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    
    # Upsampling 3: 32x32 -> 64x64
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 64x64
    x = layers.Concatenate()([x, skips[1]])
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    
    # Upsampling 4: 64x64 -> 128x128
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 128x128
    x = layers.Concatenate()([x, skips[0]])
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    
    # Upsampling 5: 128x128 -> 256x256
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # Now 256x256
    
    # Add a few more convolutions to refine features
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    
    # Final layer
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=bce_dice_loss,
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    return model

# Main training process with validation
def train_model():
    # Create debug directory
    os.makedirs("debug_masks", exist_ok=True)
    
    # Prepare dataset with validation of image-annotation pairs
    train_generator = SegmentationDataGenerator(TRAIN_IMG_DIR, TRAIN_ANN_DIR, batch_size=8, is_training=True)
    val_generator = SegmentationDataGenerator(VAL_IMG_DIR, VAL_ANN_DIR, batch_size=8, is_training=False)
    
    # Check if generators returned valid data
    print("Checking training data...")
    train_images, train_masks = train_generator[0]
    print(f"Training batch shape: {train_images.shape}, {train_masks.shape}")
    print(f"Training mask values: min={np.min(train_masks)}, max={np.max(train_masks)}")
    
    # Build model with improved architecture
    model = build_unet_model()
    print(model.summary())
    
    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_segmentation_model.h5", 
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        )
    ]
    
    # Train model with validation
    history = model.fit(
        train_generator, 
        validation_data=val_generator, 
        epochs=100,  # Train longer
        callbacks=callbacks
    )
    
    # Save final model
    model.save("final_segmentation_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.savefig('training_history.png')
    plt.close()
    
    return model

if __name__ == "__main__":
    train_model()