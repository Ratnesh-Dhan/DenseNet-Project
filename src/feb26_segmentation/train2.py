import tensorflow as tf
import os, json
import matplotlib.pyplot as plt

# ==== GPU SETUP - OPTIMIZED FOR YOUR RTX 3060 Ti ====
print("===== GPU CONFIGURATION DIAGNOSTICS =====")
print(f"TensorFlow version: {tf.__version__}")
print(f"Cuda visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Set memory growth - better approach for your RTX 3060 Ti
physical_devices = tf.config.list_physical_devices()
print(f"All physical devices available to TensorFlow: {physical_devices}")

physical_devices_gpu = tf.config.list_physical_devices('GPU')
if len(physical_devices_gpu) > 0:
    print(f"Found {len(physical_devices_gpu)} GPU(s): {physical_devices_gpu}")
    try:
        # Better memory management for RTX 3060 Ti
        for device in physical_devices_gpu:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for GPU: {device}")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found. Let's verify CUDA installation:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Forced CPU mode")

# Adjust for PASCAL VOC 2012 dataset
try:
    with open("../../Datasets/PASCAL VOC 2012/meta.json", "r") as file:
        data = json.load(file)
    NUM_CLASSES = len(data['classes'])
    print(f"Loaded metadata successfully. Number of classes: {NUM_CLASSES}")
except Exception as e:
    print(f"Error loading metadata: {e}")
    NUM_CLASSES = 21  # Default for PASCAL VOC
    print(f"Using default number of classes: {NUM_CLASSES}")

# Constants - Reduced for better performance on 8GB RTX 3060 Ti
IMG_SIZE = 448  # Reduced from 512 to lower memory requirements
BATCH_SIZE = 4  # Further reduced batch size for your GPU
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
    mask = tf.cast(mask, tf.int32)

    # Convert mask to one-hot encoding
    mask = tf.one_hot(tf.squeeze(mask), NUM_CLASSES)
    
    return image, mask

def create_dataset(tfrecord_path):
    if not os.path.exists(tfrecord_path):
        print(f"Warning: TFRecord file not found: {tfrecord_path}")
        print(f"Checking absolute path: {os.path.abspath(tfrecord_path)}")
        # Create an empty dataset for testing
        return tf.data.Dataset.from_tensors(
            (tf.zeros((IMG_SIZE, IMG_SIZE, 3)), tf.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES)))
        ).repeat().batch(BATCH_SIZE)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=100)  # Reduced buffer size
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    #dataset = dataset.repeat()  # Moved repeat after batch to reduce memory usage
    return dataset

def create_mask_rcnn_model():
    # Using a more memory-efficient approach
    with tf.device('/CPU:0'):  # Load model on CPU first
        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    
    # Freeze early layers to speed up training and reduce memory usage
    for layer in backbone.layers[:100]:
        layer.trainable = False
        
    print(f"Backbone output shape: {backbone.output.shape}")
    
    # Fixed upsampling path with lower filter counts
    x = backbone.output
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)  # Reduced from 256
    
    # Upsampling path with reduced number of filters to save memory
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(8, (4, 4), strides=2, padding='same', activation='relu')(x)  # Reduced from 16
    
    # Final layer with softmax activation
    mask_output = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='final_output')(x)
    
    print(f"Model output shape (should be (None, {IMG_SIZE}, {IMG_SIZE}, {NUM_CLASSES})): {mask_output.shape}")
    
    model = tf.keras.Model(inputs=backbone.input, outputs=mask_output)
    return model

def train_model():
    # Using absolute paths can help with file access issues
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_tfrecord = os.path.abspath(os.path.join(current_dir, '../../Datasets/PASCAL VOC 2012/train/tfrecords/train.tfrecord'))
    val_tfrecord = os.path.abspath(os.path.join(current_dir, '../../Datasets/PASCAL VOC 2012/train/tfrecords/val.tfrecord'))
    
    print("Creating datasets...")
    print(f"Train TFRecord path: {train_tfrecord}")
    print(f"Validation TFRecord path: {val_tfrecord}")
    
    train_dataset = create_dataset(train_tfrecord)
    val_dataset = create_dataset(val_tfrecord)
    
    print("Building model...")
    model = create_mask_rcnn_model()
    
    # Display model summary without printing all layers (saves console space)
    model.summary(line_length=100, expand_nested=False)
    
    # Compile with mixed precision for better GPU performance
    try:
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),  # Using legacy optimizer for better compatibility
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get the dataset size for steps_per_epoch
    print("Calculating dataset sizes...")
    try:
        if os.path.exists(train_tfrecord):
            train_size = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecord))
        else:
            train_size = 152  # Default
        
        if os.path.exists(val_tfrecord):
            val_size = sum(1 for _ in tf.data.TFRecordDataset(val_tfrecord))
        else:
            val_size = 38  # Default
    except Exception as e:
        print(f"Error calculating dataset sizes: {e}")
        train_size = 152  # Default
        val_size = 38     # Default
    
    print(f"Training set: {train_size} examples")
    print(f"Validation set: {val_size} examples")

    # Calculate steps
    steps_per_epoch = max(1, train_size // BATCH_SIZE)
    validation_steps = max(1, val_size // BATCH_SIZE)

    # Add more checkpoints to prevent loss of progress if training crashes
    os.makedirs('./checkpoints', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_mask_rcnn_model.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            './checkpoints/model_epoch_{epoch:02d}.keras',
            save_freq='epoch',  # Save every epoch
            save_weights_only=True  # Save only weights to reduce disk space
        ),
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
            update_freq='epoch'
        )
    ]
    
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
    
    try:
        # Use more explicit error handling for the fit process
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
            max_queue_size=10,  # Reduce queue size
            workers=4,  # Limit number of workers
            use_multiprocessing=True
        )
        return model, history
    except tf.errors.ResourceExhaustedError as e:
        print(f"GPU memory exhausted: {e}")
        print("Try reducing IMG_SIZE or BATCH_SIZE further")
        raise
    except Exception as e:
        print(f"Training error: {e}")
        raise

if __name__ == "__main__":
    # Train the model with better error handling
    try:
        print("Starting training process...")
        model, history = train_model()
        
        # Save model in TF format which is more reliable
        tf.saved_model.save(model, 'mask_rcnn_model_tf')
        print("Model saved in TensorFlow format")
        
        # Also save in Keras format as backup
        try:
            model.save('mask_rcnn_model.keras')
            print("Model also saved in Keras format")
        except Exception as e:
            print(f"Could not save in Keras format: {e}")
        
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
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()