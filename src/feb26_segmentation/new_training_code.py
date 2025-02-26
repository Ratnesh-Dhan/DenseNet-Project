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

# Set memory limit for GPU - IMPORTANT FOR RTX 3060 Ti
if len(physical_devices) > 0:
    try:
        # Limit GPU memory to 6GB (leaves 2GB for system)
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]
        )
        print("GPU memory limited to 6GB")
    except Exception as e:
        print(f"Error setting memory limit: {e}")

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

# Constants - FURTHER REDUCED FOR MEMORY OPTIMIZATION
IMG_SIZE = 320  # Reduced from 384 to save memory
BATCH_SIZE = 1  # Reduced to 1 to ensure it fits in memory
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
    image = tf.cast(image, tf.float16) / 255.0  # Use float16 for mixed precision
    
    # Decode mask
    mask = tf.image.decode_png(features['image/segmentation/class/encoded'], channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')
    mask = tf.cast(mask, tf.uint8)  # Use uint8 instead of int32 to save memory

    # Convert mask to one-hot encoding
    mask = tf.one_hot(tf.squeeze(mask), NUM_CLASSES)
    mask = tf.cast(mask, tf.float16)  # Convert to float16 for mixed precision
    
    return image, mask


def create_dataset(tfrecord_path):
    if not os.path.exists(tfrecord_path):
        print(f"Warning: TFRecord file not found: {tfrecord_path}")
        # Create a dummy dataset for testing
        dummy_data = (tf.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float16), 
                     tf.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=tf.float16))
        return tf.data.Dataset.from_tensors(dummy_data).repeat(3).batch(BATCH_SIZE)
    
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
        dummy_data = (tf.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=tf.float16), 
                     tf.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=tf.float16))
        return tf.data.Dataset.from_tensors(dummy_data).repeat(3).batch(BATCH_SIZE)
    
    # Create the actual dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=50)  # Reduced buffer size to save memory
    # dataset = dataset.repeat()  # Repeat the dataset for multiple epochs
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
    """Create an even lighter model to reduce memory usage"""
    # Use MobileNetV2 with even smaller alpha to reduce memory usage
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=0.5  # Reduced from 0.75 to 0.5 (50% channels)
    )
    
    # Freeze more layers to reduce training memory
    for layer in backbone.layers[:120]:
        layer.trainable = False
        
    # Print the backbone output shape for reference
    print(f"Backbone output shape: {backbone.output.shape}")
    
    # Even more simplified decoder path with fewer filters
    x = backbone.output
    
    # Use a smaller bottleneck
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    # Upsampling path - fewer filters in each layer
    x = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(48, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same', activation='relu')(x)
    
    # If needed, add one more upsampling to get to target size
    if IMG_SIZE > 256:
        x = tf.keras.layers.Conv2DTranspose(8, (4, 4), strides=2, padding='same', activation='relu')(x)
    
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
    
    # Use a memory-efficient checkpoint callback - FIXED: Was uncommented in callback list but commented in definition
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'best_model_checkpoint.weights.h5',
        save_best_only=True,
        save_weights_only=True,  # Only save weights to reduce file size
        monitor='val_loss'
    )
    
    # Compile with mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate steps - reduce steps to prevent long epochs
    steps_per_epoch = max(1, min(train_size // BATCH_SIZE, 30))  # Reduced from 50 to 30
    validation_steps = max(1, min(val_size // BATCH_SIZE, 10))   # Reduced from 20 to 10
    
    print(f"Using steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")
    
    # Create a custom callback to clear memory after each epoch
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            import gc
            gc.collect()
            tf.keras.backend.clear_session()
            print("Memory cleaned up at epoch end")
    
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
        ),
        MemoryCleanupCallback()  # Add custom callback to clean memory
    ]
    
    # Memory cleanup before training
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    try:
        # Manual checkpointing to handle memory issues
        best_val_loss = float('inf')
        model_path = 'mask_model_weights.weights.h5'
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            
            # Train for one epoch
            history_train = model.fit(
                train_dataset,
                epochs=1,
                steps_per_epoch=steps_per_epoch,
                verbose=1
            )
            
            # Validate
            history_val = model.evaluate(
                val_dataset,
                steps=validation_steps,
                verbose=1
            )
            
            val_loss = history_val[0]
            val_acc = history_val[1]
            
            print(f"Epoch {epoch+1}/{EPOCHS} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(model_path)
                print(f"Model improved, weights saved to {model_path}")
            
            # Clear memory
            gc.collect()
            tf.keras.backend.clear_session()
            
            # Rebuild model and reload weights if needed
            if (epoch + 1) % 5 == 0:
                print("Rebuilding model to prevent memory fragmentation...")
                del model
                gc.collect()
                tf.keras.backend.clear_session()
                
                model = create_lighter_model()
                model.compile(
                    optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                if os.path.exists(model_path):
                    model.load_weights(model_path)
                    print(f"Weights loaded from {model_path}")
                    
                print("Model rebuilt successfully")
        
        print("Training completed successfully!")
        return model, None
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # Train the model
    try:
        # Set environment variable to limit memory growth
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        
        # Explicitly run garbage collection before starting
        import gc
        gc.collect()
        
        model, history = train_model()
        
        if model is not None:
            print("Training completed!")
            
            # Save final model weights
            model.save_weights('final_mask_model_weights.h5')
            model.save('final_mask_model_weights.keras')
            print("Final weights saved to final_mask_model_weights.keras")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()