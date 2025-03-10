import os
import json, cv2
import base64
import zlib, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
import io, matplotlib.pyplot as plt

# Paths to dataset
TRAIN_IMG_DIR = "../../Datasets/PASCAL VOC 2012/train/img"
TRAIN_ANN_DIR = "../../Datasets/PASCAL VOC 2012/train/ann"
VAL_IMG_DIR = "../../Datasets/PASCAL VOC 2012/val/img"
VAL_ANN_DIR = "../../Datasets/PASCAL VOC 2012/val/ann"

# Image size compatible with EfficientNetB7
IMG_SIZE = (320, 320)

def augment_image_and_mask(image, mask):
    if random.random() > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if random.random() > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    if random.random() > 0.5:
        image = tf.image.adjust_brightness(image, delta=random.uniform(-0.2, 0.2))

    return image, mask

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
    
    # Normalize mask to 0 or 1 (binary mask)
    resized_mask = (resized_mask > 0).astype(np.float32)
    
    # Return the resized image and mask
    return augment_image_and_mask(resized_image, np.expand_dims(resized_mask, axis=-1))

# Data generator class
class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, img_dir, ann_dir, batch_size=8, shuffle=True):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.batch_size = batch_size
        self.image_filenames = [f for f in os.listdir(img_dir) if os.path.exists(os.path.join(ann_dir, f + ".json"))]
        # Settings for shuffle
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.image_filenames) # Initial shuffling

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
                batch_images.append(image)
                batch_masks.append(mask)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                continue
                
        if not batch_images:  # Handle empty batch case
            # Create a dummy sample to avoid training errors
            dummy_image = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            dummy_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)
            return np.array([dummy_image]), np.array([dummy_mask])
            
        return np.array(batch_images), np.array(batch_masks)
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.image_filenames)

# Define IoU metric
def iou_score(y_true, y_pred):
    # Convert predictions to binary
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # Add a small epsilon to avoid division by zero
    epsilon = tf.keras.backend.epsilon()
    iou = (intersection + epsilon) / (union + epsilon)
    
    return tf.reduce_mean(iou)

# Fixed Model using EfficientNetB7 as backbone with proper decoder for dimension matching
# def build_model():
#     # Input size for EfficientNetB7
#     inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
#     # Use EfficientNetB7 as the encoder backbone
#     base_model = keras.applications.EfficientNetB7(
#         include_top=False, 
#         input_tensor=inputs,
#         weights='imagenet'
#     )
    
#     # Freeze the base model to prevent it from being updated during initial training
#     base_model.trainable = False
    
#     # Get the output from the backbone
#     encoder_output = base_model.output
    
#     # Print model architecture to understand layer dimensions
#     # Use this for debugging but comment out for production
#     # for layer in base_model.layers:
#     #     print(f"Layer: {layer.name}, Output shape: {layer.output_shape}")
    
#     # Decoder pathway - using only the encoder output without skip connections initially
#     # to simplify and fix the dimension issue
    
#     # First upsampling block
#     x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(encoder_output)
#     x = layers.BatchNormalization()(x)
#     x = layers.UpSampling2D((2, 2))(x)
    
#     # Second upsampling block
#     x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.UpSampling2D((2, 2))(x)
    
#     # Third upsampling block
#     x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.UpSampling2D((2, 2))(x)
    
#     # Fourth upsampling block
#     x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.UpSampling2D((2, 2))(x)
    
#     # Fifth upsampling block - adjust based on the actual output dimensions needed
#     x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.UpSampling2D((2, 2))(x)
    
#     # Add additional upsampling blocks if needed to reach the desired output size
#     # For 320x320 we need to make sure we reach that dimension
#     if IMG_SIZE[0] == 320:
#         # This checks if we need another upsampling for 320x320
#         # The EfficientNetB7 typically downsamples by a factor of 32, so 320/32 = 10
#         # We might need 6 upsampling operations in total for 320x320
#         x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.UpSampling2D((2, 2))(x)
    
#     # Final convolution to get the output mask
#     outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
    
#     # Create the model
#     model = keras.Model(inputs=inputs, outputs=outputs)
    
#     # Compile the model
#     optimizer = keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)
#     model.compile(
#         optimizer=optimizer,
#         loss="binary_crossentropy",
#         metrics=["accuracy", iou_score]
#     )
    
#     return model

def build_model():
    # Input size for EfficientNetB7
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Use EfficientNetB7 as the encoder backbone
    base_model = keras.applications.EfficientNetB7(
        include_top=False, 
        input_tensor=inputs,
        weights='imagenet'
    )
    
    # Freeze the base model to prevent it from being updated during initial training
    base_model.trainable = False
    
    # Get the output from the backbone
    encoder_output = base_model.output
    
    # Decoder pathway - using only the encoder output without skip connections initially
    # to simplify and fix the dimension issue
    
    # First upsampling block
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(encoder_output)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Second upsampling block
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Third upsampling block
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Fourth upsampling block
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Fifth upsampling block - adjust based on the actual output dimensions needed
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # REMOVED: The extra upsampling block that was causing the dimensions to be 640x640
    # If IMG_SIZE[0] == 320:
    #     x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.UpSampling2D((2, 2))(x)
    
    # Final convolution to get the output mask
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
    
    # Add explicit resize to ensure output dimensions match input dimensions
    outputs = tf.keras.layers.Resizing(IMG_SIZE[0], IMG_SIZE[1])(outputs)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", iou_score]
    )
    
    return model

# Advanced model with proper skip connections
# Note: This alternative architecture requires visual inspection of the tensor dimensions
# to ensure skip connections work correctly
def build_unet_model():
    # Input size for EfficientNetB7
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Get the pre-trained EfficientNetB7 model
    base_model = keras.applications.EfficientNetB7(
        include_top=False,
        input_tensor=inputs,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create a dictionary to store the outputs of the layers we want to use as skip connections
    # Note: We need to inspect the model to get the correct layer names and dimensions
    # These are approximate and may need adjustment
    skips = {}
    
    # Define which layers to extract for skip connections
    # We'll need to print the model summary to find the right layers
    # These values are examples and should be adjusted
    skips["block2d_add"] = base_model.get_layer("block2d_add").output        # Earlier feature map
    skips["block3d_add"] = base_model.get_layer("block3d_add").output        # Mid-level feature map
    skips["block5i_add"] = base_model.get_layer("block5i_add").output        # Higher-level feature map
    
    # Get the bottleneck features from the encoder
    x = base_model.output
    
    # Start the decoder path
    # The dimensions below are approximations and should be verified
    
    # Decoder Block 1
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample
    
    # Decoder Block 2 with skip connection
    # Resize the skip connection to match the upsampled features if needed
    skip1 = layers.Conv2D(512, (1, 1), padding="same")(skips["block5i_add"])
    skip1_shape = tf.shape(skip1)[1:3]
    x_shape = tf.shape(x)[1:3]
    
    # Resize if dimensions don't match
    if tf.not_equal(skip1_shape[0], x_shape[0]) or tf.not_equal(skip1_shape[1], x_shape[1]):
        skip1 = tf.image.resize(skip1, x_shape)
    
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample
    
    # Decoder Block 3 with skip connection
    skip2 = layers.Conv2D(256, (1, 1), padding="same")(skips["block3d_add"])
    skip2_shape = tf.shape(skip2)[1:3]
    x_shape = tf.shape(x)[1:3]
    
    # Resize if dimensions don't match
    if tf.not_equal(skip2_shape[0], x_shape[0]) or tf.not_equal(skip2_shape[1], x_shape[1]):
        skip2 = tf.image.resize(skip2, x_shape)
    
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample
    
    # Decoder Block 4 with skip connection
    skip3 = layers.Conv2D(128, (1, 1), padding="same")(skips["block2d_add"])
    skip3_shape = tf.shape(skip3)[1:3]
    x_shape = tf.shape(x)[1:3]
    
    # Resize if dimensions don't match
    if tf.not_equal(skip3_shape[0], x_shape[0]) or tf.not_equal(skip3_shape[1], x_shape[1]):
        skip3 = tf.image.resize(skip3, x_shape)
    
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample
    
    # Continue upsampling to reach the target size
    # We need to get from the current size to IMG_SIZE
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample
    
    # One more upsampling to reach 320x320
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)  # Upsample
    
    # Final 1x1 convolution to get the segmentation mask
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
    
    # Ensure the output dimensions match the input dimensions
    outputs_shape = tf.shape(outputs)[1:3]
    if tf.not_equal(outputs_shape[0], IMG_SIZE[0]) or tf.not_equal(outputs_shape[1], IMG_SIZE[1]):
        outputs = tf.image.resize(outputs, [IMG_SIZE[0], IMG_SIZE[1]])
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", iou_score]
    )
    
    return model

# Check if the data directories exist
print(f"Train image directory exists: {os.path.exists(TRAIN_IMG_DIR)}")
print(f"Train annotation directory exists: {os.path.exists(TRAIN_ANN_DIR)}")
print(f"Validation image directory exists: {os.path.exists(VAL_IMG_DIR)}")
print(f"Validation annotation directory exists: {os.path.exists(VAL_ANN_DIR)}")

# Prepare dataset with smaller batch size due to EfficientNetB7's memory requirements
train_generator = SegmentationDataGenerator(TRAIN_IMG_DIR, TRAIN_ANN_DIR, batch_size=2)  # Reduced from 4 to 2
val_generator = SegmentationDataGenerator(VAL_IMG_DIR, VAL_ANN_DIR, batch_size=2)  # Reduced from 4 to 2

# Print sample counts
print(f"Number of training samples: {len(train_generator) * train_generator.batch_size}")
print(f"Number of validation samples: {len(val_generator) * val_generator.batch_size}")

# Let's use the simplified model for now
# For advanced model, uncomment the line below and comment out the line above after inspection
model = build_model()
# model = build_unet_model()  # Use this for the advanced model with skip connections

# Print model summary to see the output dimensions of each layer
model.summary()

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    verbose=1,
    restore_best_weights=True,
    min_delta=0.001
)

model_checkpoint = ModelCheckpoint(
    filepath='best_segmentation_model_b7.h5',
    monitor='val_iou_score',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

# Train model with callbacks
try:
    print("Starting initial training phase with frozen backbone...")
    
    # Initial training with frozen backbone
    history_initial = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Initial training successful! Now fine-tuning the model...")
    
    # Unfreeze the backbone model for fine-tuning
    for layer in model.layers:
        layer.trainable = True
    
    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy", iou_score]
    )
    
    # Continue training with all layers unfrozen
    history_finetuning = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=60,
        initial_epoch=5,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = {}
    for key in history_initial.history:
        history[key] = history_initial.history[key] + history_finetuning.history[key]
    
    # Save final model
    model.save("segmentation_model_b7_final.h5")
    
    # Plot training history
    plt.figure(figsize=(18, 6))
    
    # Plot training & validation loss
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation IoU
    plt.subplot(1, 3, 3)
    plt.plot(history['iou_score'])
    plt.plot(history['val_iou_score'])
    plt.title('Model IoU Score')
    plt.ylabel('IoU Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history_b7.png')
    plt.show()
    
except Exception as e:
    print(f"Training error: {str(e)}")
    print("Please check your data and model configuration.")