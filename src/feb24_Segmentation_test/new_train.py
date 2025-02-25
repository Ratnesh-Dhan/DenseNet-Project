import tensorflow as tf
import numpy as np
import os, json
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors

with open("../../Datasets/testDataset/meta.json", "r") as file:
        data = json.load(file)
# Constants
IMG_SIZE = 256 #512
BATCH_SIZE = 2
NUM_CLASSES = len(data['classes']) #number of classes
EPOCHS = 10


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
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat().batch(BATCH_SIZE) # Leaving repeat() without value will make it run indefenietly . with value like "repeat(2)" will make dataset run for 2 times
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_mask_rcnn_model():
    # backbone = tf.keras.applications.MobileNetV2(
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    x = backbone.output
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    # Upsample to match the target mask size (512x512)

    # x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 64x64
    # x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 128x128
    # x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 256x256
    # x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 512x512

    x = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(x) #64*64
    x = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=2, activation='relu')(x) #64*64
    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, activation='relu')(x) #128*128
    x = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=2, activation='relu')(x) #256*256
    x = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=2, activation='relu')(x) #512*512
    # x = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(x) #512*512

    mask_output = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(x)
    
    model = tf.keras.Model(inputs=backbone.input, outputs=mask_output)
    return model


def train_model():
    train_tfrecord = '../../Datasets/testDataset/tfrecords/train.tfrecord'
    val_tfrecord = '../../Datasets/testDataset/tfrecords/val.tfrecord'
    
    train_dataset = create_dataset(train_tfrecord)
    val_dataset = create_dataset(val_tfrecord)
    
    model = create_mask_rcnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_mask_rcnn_model.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        # tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=10,
        #     restore_best_weights=True
        # )
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return model, history


if __name__ == "__main__":
    model, history = train_model()
    model.save('mask_rcnn_model.keras')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')

