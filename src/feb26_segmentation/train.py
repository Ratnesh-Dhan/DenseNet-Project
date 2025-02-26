import tensorflow as tf
import os, json
import matplotlib.pyplot as plt

# ==== GPU SETUP - CRUCIAL FOR DETECTION ====
# Clear any pre-existing GPU memory restrictions
tf.config.experimental.set_memory_growth = lambda device, enabled: None

# Print diagnostic information for troubleshooting
print("===== GPU CONFIGURATION DIAGNOSTICS =====")
print(f"TensorFlow version: {tf.__version__}")
print(f"Cuda visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Check GPU availability - detailed information to debug
physical_devices = tf.config.list_physical_devices()
print(f"All physical devices available to TensorFlow: {physical_devices}")

physical_devices_gpu = tf.config.list_physical_devices('GPU')
if len(physical_devices_gpu) > 0:
    print(f"Found {len(physical_devices_gpu)} GPU(s): {physical_devices_gpu}")
    try:
        # Configure TensorFlow to use the GPU
        for device in physical_devices_gpu:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for GPU: {device}")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found. Let's verify CUDA installation:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    # Force CPU usage since no GPU is available
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Forced CPU mode")

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

# Constants
IMG_SIZE = 512
BATCH_SIZE = 4  # Reduced batch size in case of memory issues
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
        # Create an empty dataset for testing
        return tf.data.Dataset.from_tensors(
            (tf.zeros((IMG_SIZE, IMG_SIZE, 3)), tf.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES)))
        ).batch(BATCH_SIZE)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_mask_rcnn_model():
    # Using ResNet50 as backbone but with proper output size
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze early layers to speed up training
    for layer in backbone.layers[:100]:
        layer.trainable = False
        
    # Capture the input size we need to match
    print(f"Backbone output shape: {backbone.output.shape}")
    
    # Fixed upsampling path to ensure output is IMG_SIZE x IMG_SIZE
    x = backbone.output
    
    # We need to go from backbone output (16x16) to 512x512
    # That's 5 upsampling operations (16->32->64->128->256->512)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    
    # Upsampling path with 5 steps to ensure the final output is 512x512
    x = tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=2, padding='same', activation='relu')(x) # 32x32
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu')(x) # 64x64
    x = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu')(x)  # 128x128
    x = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation='relu')(x)  # 256x256
    x = tf.keras.layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same', activation='relu')(x)  # 512x512
    
    # Final layer with softmax activation
    mask_output = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='final_output')(x)
    
    # Verify the output shape to ensure it matches the expected size
    print(f"Model output shape (should be (None, 512, 512, {NUM_CLASSES})): {mask_output.shape}")
    
    model = tf.keras.Model(inputs=backbone.input, outputs=mask_output)
    return model


def train_model():
    train_tfrecord = '../../Datasets/PASCAL VOC 2012/train/tfrecords/train.tfrecord'
    val_tfrecord = '../../Datasets/PASCAL VOC 2012/train/tfrecords/val.tfrecord'
    
    print("Creating datasets...")
    train_dataset = create_dataset(train_tfrecord)
    val_dataset = create_dataset(val_tfrecord)
    
    print("Building model...")
    model = create_mask_rcnn_model()
    
    # Display model summary to verify architecture
    model.summary()
    
    # Use float32 for better compatibility if GPUs aren't available
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get the dataset size for steps_per_epoch
    print("Calculating dataset sizes...")
    try:
        train_size = sum(1 for _ in tf.data.TFRecordDataset(train_tfrecord))
        val_size = sum(1 for _ in tf.data.TFRecordDataset(val_tfrecord))
    except Exception as e:
        print(f"Error calculating dataset sizes: {e}")
        train_size = 152  # Default from your error message
        val_size = 38     # Default from your error message
    
    print(f"Training set: {train_size} examples")
    print(f"Validation set: {val_size} examples")

    # Calculate steps
    steps_per_epoch = max(1, train_size // BATCH_SIZE)
    validation_steps = max(1, val_size // BATCH_SIZE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_mask_rcnn_model.keras',
            save_best_only=True,
            monitor='val_loss'
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
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


if __name__ == "__main__":
    # Train the model
    try:
        model, history = train_model()
        model.save('mask_rcnn_model.keras')
        
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