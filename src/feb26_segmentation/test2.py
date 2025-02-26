import tensorflow as tf
import os, json
import matplotlib.pyplot as plt

# ==== GPU SETUP - CRUCIAL FOR DETECTION ====
# Configure GPU memory growth to avoid taking all memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
else:
    print("No GPU found. Forcing CPU mode.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enable mixed precision to reduce memory usage
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision policy set to: mixed_float16")
except Exception as e:
    print(f"Could not set mixed precision: {e}")

# Print diagnostic information
print(f"TensorFlow version: {tf.__version__}")
print(f"Cuda visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Load your metadata
try:
    with open("../../Datasets/PASCAL VOC 2012/meta.json", "r") as file:
        data = json.load(file)
    NUM_CLASSES = len(data['classes'])
    print(f"Loaded metadata successfully. Number of classes: {NUM_CLASSES}")
except Exception as e:
    print(f"Error loading metadata: {e}")
    NUM_CLASSES = 21  # Default for PASCAL VOC
    print(f"Using default number of classes: {NUM_CLASSES}")

# Constants - REDUCED FOR MEMORY OPTIMIZATION
IMG_SIZE = 384  # Reduced from 512 to save memory
BATCH_SIZE = 2  # Reduced from 4 based on memory analysis
EPOCHS = 100


def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/segmentation/class/encoded': tf.io.FixedLenFeature([], tf.string)
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode image
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    
    # Decode mask
    mask = tf.image.decode_png(features['image/segmentation/class/encoded'], channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')
    mask = tf.cast(mask, tf.uint8)  # Use uint8 instead of int32 to save memory

    # Convert mask to one-hot encoding
    mask = tf.one_hot(tf.squeeze(mask), NUM_CLASSES)
    
    return image, mask


def create_dataset(tfrecord_path):
    if not os.path.exists(tfrecord_path):
        print(f"Warning: TFRecord file not found: {tfrecord_path}")
        # Create a dummy dataset for testing
        dummy_data = (tf.zeros((IMG_SIZE, IMG_SIZE, 3)), tf.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES)))
        return tf.data.Dataset.from_tensors(dummy_data).repeat().batch(BATCH_SIZE)
    
    try:
        # Verify the TFRecord file is readable
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        # Try reading one example to verify file is valid
        iterator = iter(dataset)
        next(iterator)
        print(f"Successfully verified TFRecord: {tfrecord_path}")
    except Exception as e:
        print(f"Error reading TFRecord {tfrecord_path}: {e}")
        # Create a dummy dataset for testing
        dummy_data = (tf.zeros((IMG_SIZE, IMG_SIZE, 3)), tf.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES)))
        return tf.data.Dataset.from_tensors(dummy_data).repeat().batch(BATCH_SIZE)
    
    # Create the actual dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=100)  # Reduced buffer size to save memory
    dataset = dataset.repeat()  # Repeat the dataset for multiple epochs
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def count_examples(tfrecord_path):
    if not os.path.exists(tfrecord_path):
        return 0
    
    try:
        # Count examples more efficiently
        count = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
        return count
    except Exception as e:
        print(f"Error counting examples: {e}")
        # Default values from your original script
        if 'train' in tfrecord_path:
            return 152
        else:
            return 38


def create_lighter_model():
    """Create a lighter model to reduce memory usage"""
    # Use MobileNetV2 instead of ResNet50 as backbone
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=0.75  # Use a smaller model (75% channels)
    )
    
    # Freeze early layers to speed up training
    for layer in backbone.layers[:100]:
        layer.trainable = False
        
    # Print the backbone output shape for reference
    print(f"Backbone output shape: {backbone.output.shape}")
    
    # Simplified decoder path with fewer filters
    x = backbone.output
    
    # Use a smaller bottleneck
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    
    # Upsampling path - more gradual to reduce memory spikes
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(96, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu')(x)
    
    # If needed, add one more upsampling to get to target size
    if IMG_SIZE > 256:
        x = tf.keras.layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same', activation='relu')(x)
    
    # Final layer with softmax activation
    mask_output = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='final_output')(x)
    
    # Verify the output shape
    model = tf.keras.Model(inputs=backbone.input, outputs=mask_output)
    print(f"Model output shape: {mask_output.shape}")
    
    return model


def train_model():
    train_tfrecord = '../../Datasets/PASCAL VOC 2012/train/tfrecords/train.tfrecord'
    val_tfrecord = '../../Datasets/PASCAL VOC 2012/train/tfrecords/val.tfrecord'
    
    # Verify and count examples in datasets
    print("Verifying and counting dataset examples...")
    train_size = count_examples(train_tfrecord)
    val_size = count_examples(val_tfrecord)
    
    print(f"Training set: {train_size} examples")
    print(f"Validation set: {val_size} examples")
    
    # Create datasets with proper repetition
    print("Creating datasets...")
    train_dataset = create_dataset(train_tfrecord)
    val_dataset = create_dataset(val_tfrecord)
    
    # Build a lighter model
    print("Building model...")
    model = create_lighter_model()
    
    # Use a memory-efficient checkpoint callback
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     'best_model_checkpoint.weights.h5',
    #     save_best_only=True,
    #     save_weights_only=True,  # Only save weights to reduce file size
    #     monitor='val_loss'
    # )
    
    # Compile with mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate steps
    steps_per_epoch = max(1, min(train_size // BATCH_SIZE, 50))  # Cap steps to avoid long epochs
    validation_steps = max(1, min(val_size // BATCH_SIZE, 20))
    
    print(f"Using steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")
    
    callbacks = [
        checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            update_freq='epoch',
            profile_batch=0  # Disable profiling to save memory
        )
    ]
    
    # Memory cleanup before training
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model weights only to save disk space
        model.save_weights('mask_model_weights.keras')
        
        # Plot and save training history
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training completed successfully!")
        
        return model, history
    
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Train the model
    try:
        model, history = train_model()
        
        if model is not None and history is not None:
            print("Training completed!")
            
            # Save history to file for later analysis
            with open('training_history.json', 'w') as f:
                history_dict = {key: [float(val) for val in history.history[key]] 
                               for key in history.history.keys()}
                json.dump(history_dict, f)
            
            print("History saved to training_history.json")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()