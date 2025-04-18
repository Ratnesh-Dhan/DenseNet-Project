import tensorflow as tf
import numpy as np, os
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors

# Constants
IMG_SIZE = 512
BATCH_SIZE = 2
NUM_CLASSES = 21
EPOCHS = 50

def parse_tfrecord(example_proto):
    """Parse TFRecord data."""
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
    
    # Convert mask to one-hot encoding
    mask = tf.one_hot(tf.squeeze(mask), NUM_CLASSES)
    
    return image, mask

def create_dataset(tfrecord_path):
    """Create dataset from TFRecord file."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_mask_rcnn_model():
    """Create simplified Mask R-CNN model architecture."""
    # Use ResNet50 as backbone
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Feature Pyramid Network
    C2 = backbone.get_layer('conv2_block3_out').output
    C3 = backbone.get_layer('conv3_block4_out').output
    C4 = backbone.get_layer('conv4_block6_out').output
    C5 = backbone.get_layer('conv5_block3_out').output
    
    # FPN layers
    P5 = tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = tf.keras.layers.Add(name="fpn_p4add")([
        tf.keras.layers.UpSampling2D(size=(2, 2))(P5),
        tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)
    ])
    P3 = tf.keras.layers.Add(name="fpn_p3add")([
        tf.keras.layers.UpSampling2D(size=(2, 2))(P4),
        tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = tf.keras.layers.Add(name="fpn_p2add")([
        tf.keras.layers.UpSampling2D(size=(2, 2))(P3),
        tf.keras.layers.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)
    ])
    
    # Mask head
    x = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2)(P2)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    mask_output = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(x)
    
    model = tf.keras.Model(inputs=backbone.input, outputs=mask_output)
    return model

def train_model():
    """Train the Mask R-CNN model using TFRecord data."""
    # Setup paths for TFRecord files
    train_tfrecord = '../../Datasets/testDataset/tfrecords/train.tfrecord'
    val_tfrecord = '../../Datasets/testDataset/tfrecords/val.tfrecord'
    
    # Create datasets
    train_dataset = create_dataset(train_tfrecord)
    val_dataset = create_dataset(val_tfrecord)
    
    # Create and compile model
    model = create_mask_rcnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_mask_rcnn_model.h5',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    # Train model
    model, history = train_model()
    
    # Save model
    model.save('mask_rcnn_model')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')