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

# Increased image size for EfficientNetB7 which works better with higher resolution
IMG_SIZE = (320, 320)  # Increased from 256x256 to better leverage EfficientNetB7

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

# Model using EfficientNetB7 as backbone with improved decoder
def build_model():
    # Input size for EfficientNetB7
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Use EfficientNetB7 as the encoder (backbone)
    base_model = keras.applications.EfficientNetB7(
        include_top=False, 
        input_tensor=inputs,
        weights='imagenet'
    )
    
    # Freeze the base model to prevent it from being updated during initial training
    base_model.trainable = False
    
    # Extract features from different layers of the encoder for skip connections
    # For EfficientNetB7, we'll use multiple layers for skip connections
    # These layer names might vary slightly - consult EfficientNetB7 architecture if needed
    skip_features = {}
    skip_features['s1'] = base_model.get_layer('block1a_project_bn').output  # early feature
    skip_features['s2'] = base_model.get_layer('block3a_project_bn').output  # mid-level feature
    skip_features['s3'] = base_model.get_layer('block5a_project_bn').output  # higher-level feature
    
    # Get the final encoder output
    encoder_output = base_model.output  # Shape will be smaller than B0 due to B7's depth
    
    # Decoder pathway with proper skip connections
    # EfficientNetB7 has a deeper network, so we need more upsampling steps
    
    # Starting with bottleneck features
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(encoder_output)
    x = layers.BatchNormalization()(x)
    
    # Upsampling and adding skip connections
    # First upsampling
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, skip_features['s3']])  # Skip connection
    
    # Second upsampling
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, skip_features['s2']])  # Skip connection
    
    # Third upsampling
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, skip_features['s1']])  # Skip connection
    
    # Continue upsampling to reach the original image size
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Final upsampling to match input size
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Add one more upsampling if needed to reach the input size
    # For 320x320 inputs with EfficientNetB7, we might need one more upsampling
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Final layer to produce the segmentation mask
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", kernel_initializer='glorot_normal')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Use a more stable optimizer with clipnorm to prevent exploding gradients
    # Reduced learning rate since B7 is more sensitive during training
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

# Build model
model = build_model()
model.summary()

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(
    monitor='val_loss',      # Monitor validation loss
    patience=15,             # Number of epochs with no improvement after which training will stop
    verbose=1,               # Print messages
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    min_delta=0.001          # Minimum change to qualify as an improvement
)

# Create a ModelCheckpoint callback to save the best model during training
model_checkpoint = ModelCheckpoint(
    filepath='best_segmentation_model_b7.h5',  # Updated filename for B7
    monitor='val_iou_score',   # Monitor validation IoU
    save_best_only=True,       # Only save the best model
    mode='max',                # The higher the IoU the better
    verbose=1                  # Print messages
)

# Add a learning rate scheduler to reduce the learning rate when training plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Define callbacks list
callbacks = [early_stopping, model_checkpoint, reduce_lr]

# Add gradient accumulation to better handle the larger model with small batch sizes
# This is a simple approach using a custom training loop
def train_with_gradient_accumulation(model, train_generator, val_generator, epochs, callbacks, accumulation_steps=4):
    # Initialize training history
    history = {'loss': [], 'accuracy': [], 'iou_score': [], 
               'val_loss': [], 'val_accuracy': [], 'val_iou_score': []}
    
    # Get compiled optimizer
    optimizer = model.optimizer
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Reset metrics
        train_loss = 0.0
        train_acc = 0.0
        train_iou = 0.0
        step_count = 0
        
        # Initialize gradients
        accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
        
        # Training loop
        for batch_idx in range(len(train_generator)):
            x_batch, y_batch = train_generator[batch_idx]
            
            # Forward pass with gradient recording
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                batch_loss = model.compiled_loss(y_batch, y_pred)
            
            # Calculate gradients
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            
            # Accumulate gradients
            accumulated_gradients = [(acum_grad + grad) for acum_grad, grad in zip(accumulated_gradients, gradients)]
            
            # Get metrics
            batch_acc = model.compiled_metrics.metrics[0](y_batch, y_pred)
            batch_iou = model.compiled_metrics.metrics[1](y_batch, y_pred)
            
            train_loss += batch_loss.numpy()
            train_acc += batch_acc.numpy()
            train_iou += batch_iou.numpy()
            step_count += 1
            
            # Apply accumulated gradients after specified number of steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_generator):
                # Normalize the gradients
                accumulated_gradients = [grad / accumulation_steps for grad in accumulated_gradients]
                
                # Apply gradients
                optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
                
                # Reset accumulated gradients
                accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
                
                print(f"Batch {batch_idx+1}/{len(train_generator)} - Loss: {batch_loss.numpy():.4f}")
        
        # Calculate average training metrics
        train_loss /= step_count
        train_acc /= step_count
        train_iou /= step_count
        
        # Validation loop
        val_loss = 0.0
        val_acc = 0.0
        val_iou = 0.0
        val_steps = 0
        
        for batch_idx in range(len(val_generator)):
            x_val, y_val = val_generator[batch_idx]
            y_pred = model(x_val, training=False)
            
            # Calculate validation loss and metrics
            batch_val_loss = model.compiled_loss(y_val, y_pred)
            batch_val_acc = model.compiled_metrics.metrics[0](y_val, y_pred)
            batch_val_iou = model.compiled_metrics.metrics[1](y_val, y_pred)
            
            val_loss += batch_val_loss.numpy()
            val_acc += batch_val_acc.numpy()
            val_iou += batch_val_iou.numpy()
            val_steps += 1
        
        val_loss /= val_steps
        val_acc /= val_steps
        val_iou /= val_steps
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f} - IoU: {train_iou:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f} - Val IoU: {val_iou:.4f}")
        
        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['iou_score'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_iou_score'].append(val_iou)
        
        # Handle callbacks (simplified implementation)
        if 'best_val_iou' not in locals() or val_iou > best_val_iou:
            best_val_iou = val_iou
            model.save('best_segmentation_model_b7.h5')
            print("Saved best model with improved IoU.")
        
        # Early stopping check (simplified)
        if 'best_val_loss' not in locals():
            best_val_loss = val_loss
            patience_counter = 0
        elif val_loss < best_val_loss - early_stopping.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return history

# Train model with callbacks - start with a few epochs to check stability
try:
    print("Starting initial training phase with frozen backbone...")
    
    # Train with frozen backbone first using gradient accumulation for stability
    history_initial = train_with_gradient_accumulation(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=5,
        callbacks=callbacks,
        accumulation_steps=4  # Accumulate gradients over 4 steps (effectively like batch size 8)
    )
    
    print("Initial training successful! Now fine-tuning the model...")
    
    # Unfreeze the backbone model for fine-tuning
    for layer in model.layers:
        layer.trainable = True
    
    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0),  # Even lower LR for B7
        loss="binary_crossentropy",
        metrics=["accuracy", iou_score]
    )
    
    # Continue training with all layers unfrozen
    history_finetuning = train_with_gradient_accumulation(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=60,  # Reduced from 100 given B7's complexity
        callbacks=callbacks,
        accumulation_steps=8  # Increased accumulation for fine-tuning
    )
    
    # Combine histories
    history = {}
    for key in history_initial:
        history[key] = history_initial[key] + history_finetuning[key]
    
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