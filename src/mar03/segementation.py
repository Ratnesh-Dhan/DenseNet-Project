import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import glob
import base64
import zlib
import io
from tqdm import tqdm

# Configuration
CONFIG = {
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 1e-4,
    'image_size': (512, 512),  # Resize images to this size
    'num_classes': 21,  # 20 classes + background for Pascal VOC
    'data_dir': '../../Datasets/PASCAL VOC 2012/train/img',  # Update this with your dataset path
    'annotation_dir': '../../Datasets/PASCAL VOC 2012/train/ann',  # Update with your annotations path
    'model_save_path': './',
    'class_mapping': {
        'neutral': 0,  # Background
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
}

def decode_bitmap(bitmap_data, origin, img_size):
    """
    Decode the bitmap data to create a mask using the same approach as the visualization code
    """
    try:
        # Decode base64 data
        bitmap_data = base64.b64decode(bitmap_data)
        
        try:
            # Try to decompress the data
            decompressed_data = zlib.decompress(bitmap_data)
            bitmap_image = Image.open(io.BytesIO(decompressed_data))
            mask = np.array(bitmap_image.convert('L'))
        except:
            # If decompression fails, use the data as is
            print("Warning: Decompression failed, using raw bitmap data")
            bitmap_image = Image.open(io.BytesIO(bitmap_data))
            mask = np.array(bitmap_image.convert('L'))
            
        # Create an empty mask with the same size as the image
        full_mask = np.zeros(img_size, dtype=np.uint8)
        
        # Get the mask dimensions
        h, w = mask.shape
        
        # Place the mask at the correct position based on origin
        # Ensure we don't go out of bounds
        y_start, x_start = origin[1], origin[0]
        y_end = min(y_start + h, img_size[0])
        x_end = min(x_start + w, img_size[1])
        
        if y_start < img_size[0] and x_start < img_size[1]:
            h_to_use = y_end - y_start
            w_to_use = x_end - x_start
            
            full_mask[y_start:y_end, x_start:x_end] = mask[:h_to_use, :w_to_use]
        
        return full_mask > 0  # Convert to boolean mask
        
    except Exception as e:
        print(f"Error decoding bitmap: {e}")
        return np.zeros(img_size, dtype=np.bool_)

def parse_annotation_file(json_path, image_size=CONFIG['image_size']):
    """
    Parse the annotation JSON file and generate a segmentation mask
    """
    img_height, img_width = image_size
    
    try:
        with open(json_path, 'r') as f:
            annotation = json.load(f)
        
        # Get original image size from annotation if available
        original_height = annotation.get('size', {}).get('height', img_height)
        original_width = annotation.get('size', {}).get('width', img_width)
        
        # Create an empty mask with background class (0)
        mask = np.zeros((original_height, original_width), dtype=np.uint8)
        
        # Process each object
        for obj in annotation.get('objects', []):
            class_title = obj.get('classTitle', 'neutral')
            class_id = CONFIG['class_mapping'].get(class_title, 0)
            
            if obj.get('geometryType') == 'bitmap':
                bitmap_data = obj.get('bitmap', {}).get('data', '')
                origin = obj.get('bitmap', {}).get('origin', [0, 0])
                
                # Get object mask and place it in the correct position
                obj_mask = decode_bitmap(bitmap_data, origin, (original_height, original_width))
                
                # Set class ID to the mask where object exists
                mask[obj_mask] = class_id
        
        # Resize mask to target size
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize(image_size, Image.NEAREST)
        mask = np.array(mask_pil)
        
        return mask
    except Exception as e:
        print(f"Error parsing annotation file {json_path}: {e}")
        return np.zeros(image_size, dtype=np.uint8)

def load_image(image_path, target_size=CONFIG['image_size']):
    """
    Load and preprocess an image
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def visualize_sample(image_path, json_path, save_path=None):
    """
    Visualize a single image and its segmentation mask for verification
    """
    # Load the image
    image = Image.open(image_path)
    image = image.convert("RGB")
    
    # Parse the annotation to get the mask
    mask = parse_annotation_file(json_path, (image.height, image.width))
    
    # Create a colormap for visualization
    cmap = plt.cm.get_cmap('tab20', CONFIG['num_classes'])
    colored_mask = cmap(mask)
    
    # Prepare the figure
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='tab20')
    plt.title("Segmentation Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(colored_mask, alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_dataset(image_paths, annotation_paths, batch_size=CONFIG['batch_size'], augment=False):
    """
    Create a TensorFlow dataset from image and annotation paths
    """
    def generator():
        for img_path, ann_path in zip(image_paths, annotation_paths):
            try:
                # Load image
                image = load_image(img_path)
                
                # Parse annotation to create mask
                mask = parse_annotation_file(ann_path)
                mask = tf.convert_to_tensor(mask, dtype=tf.int32)
                
                # One-hot encode the mask
                one_hot_mask = tf.one_hot(mask, depth=CONFIG['num_classes'])
                
                yield image, one_hot_mask
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(CONFIG['image_size'][0], CONFIG['image_size'][1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(CONFIG['image_size'][0], CONFIG['image_size'][1], CONFIG['num_classes']), dtype=tf.float32)
        )
    )
    
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def augment_data(image, mask):
    """
    Apply data augmentation to the image and mask
    """
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Random brightness adjustment (apply only to image, not mask)
    image = tf.image.random_brightness(image, 0.1)
    
    # Random contrast adjustment (apply only to image, not mask)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    
    # Ensure the image values are in the range [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def build_unet_model(input_shape, num_classes):
    """
    Build a U-Net model for segmentation
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bridge
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    
    # Decoder
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    merge6 = layers.Concatenate()([drop4, up6])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.Concatenate()([conv3, up7])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = layers.Concatenate()([conv2, up8])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = layers.Concatenate()([conv1, up9])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for evaluating segmentation performance
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice loss for training segmentation models
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Combined loss: categorical cross-entropy + dice loss
    """
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return ce_loss + dice

def visualize_predictions(model, dataset, num_samples=5, save_dir='predictions'):
    """
    Visualize model predictions
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (image_batch, mask_batch) in enumerate(dataset.take(num_samples)):
        if i >= num_samples:
            break
        
        pred_mask = model.predict(image_batch)
        
        for j in range(min(1, len(image_batch))):
            # Get the first image and mask from the batch
            img = image_batch[j].numpy()
            true_mask = tf.argmax(mask_batch[j], axis=-1).numpy()
            pred_mask_idx = tf.argmax(pred_mask[j], axis=-1).numpy()
            
            # Create a figure
            plt.figure(figsize=(15, 5))
            
            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f"Image")
            plt.axis('off')
            
            # Plot true mask
            plt.subplot(1, 3, 2)
            plt.imshow(true_mask, cmap='tab20')
            plt.title(f"True Mask")
            plt.axis('off')
            
            # Plot predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask_idx, cmap='tab20')
            plt.title(f"Predicted Mask")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{i}_{j}.png'))
            plt.close()

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create necessary directories
    os.makedirs(CONFIG['model_save_path'], exist_ok=True)
    
    # Get image and annotation paths
    image_dir = os.path.join(CONFIG['data_dir'], 'JPEGImages')
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    
    annotation_paths = []
    valid_image_paths = []
    
    print("Finding matching image-annotation pairs...")
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path).split('.')[0]
        ann_path = os.path.join(CONFIG['annotation_dir'], f"{img_name}.jpg.json")
        if os.path.exists(ann_path):
            annotation_paths.append(ann_path)
            valid_image_paths.append(img_path)
    
    print(f"Found {len(valid_image_paths)} images with annotations")
    
    # Verify the first few samples to ensure masks are working correctly
    if len(valid_image_paths) > 0:
        print("Visualizing a few samples to verify mask generation...")
        os.makedirs('sample_visualizations', exist_ok=True)
        
        for i in range(min(3, len(valid_image_paths))):
            visualize_sample(
                valid_image_paths[i], 
                annotation_paths[i],
                save_path=f'sample_visualizations/sample_{i}.png'
            )
        print("Sample visualizations saved to 'sample_visualizations' directory")
    
    # Split data into train and validation sets
    indices = np.arange(len(valid_image_paths))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_img_paths = [valid_image_paths[i] for i in train_indices]
    train_ann_paths = [annotation_paths[i] for i in train_indices]
    
    val_img_paths = [valid_image_paths[i] for i in val_indices]
    val_ann_paths = [annotation_paths[i] for i in val_indices]
    
    print(f"Training on {len(train_img_paths)} images, validating on {len(val_img_paths)} images")
    
    # Create datasets
    train_dataset = create_dataset(train_img_paths, train_ann_paths, augment=True)
    val_dataset = create_dataset(val_img_paths, val_ann_paths, augment=False)
    
    # Build model
    model = build_unet_model((CONFIG['image_size'][0], CONFIG['image_size'][1], 3), CONFIG['num_classes'])
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    
    # Model summary
    model.summary()
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_save_path'], 'best_model.h5'),
            save_best_only=True,
            monitor='val_dice_coefficient',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(CONFIG['model_save_path'], 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=CONFIG['epochs'],
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(CONFIG['model_save_path'], 'final_model.h5'))
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coefficient'], label='Training Dice')
    plt.plot(history.history['val_dice_coefficient'], label='Validation Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['model_save_path'], 'training_history.png'))
    plt.close()
    
    # Visualize some predictions
    print("Generating visualization of predictions...")
    visualize_predictions(model, val_dataset, save_dir=os.path.join(CONFIG['model_save_path'], 'predictions'))
    
    print("Training completed successfully!")
    print(f"Model saved to {CONFIG['model_save_path']}")

if __name__ == "__main__":
    main()