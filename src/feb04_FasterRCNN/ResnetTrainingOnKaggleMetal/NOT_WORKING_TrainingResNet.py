# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:15:56 2025

Training our own resnet Model with kaggle Metal surface dataset

THIS CODE IS NOT WORKING . BECAUSE OF FOLDER STRUCTURE DEFINED

@author: NDT Lab
"""

import tensorflow as tf
import os
# import cv2
# import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
# import multiprocessing

# Set TensorFlow to use CPU optimizations
tf.config.threading.set_intra_op_parallelism_threads(12)  # Your CPU threads
tf.config.threading.set_inter_op_parallelism_threads(6)   # Your CPU cores

# Define model architecture with ResNet50
def create_model():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)  # ResNet's standard input size
    )
    
    # Freeze early layers for transfer learning
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')  # 6 classes for defects
    ])
    
    return model

# Data preprocessing function
def preprocess_data(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

# Create data pipeline optimized for CPU
def create_dataset(images, labels, batch_size=32, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = (dataset
        .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
    
    return dataset

# Training function with model saving
def train_model(train_dataset, val_dataset, epochs=50):
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for model saving and early stopping
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        workers=6,  # Number of CPU cores
        use_multiprocessing=True
    )
    
    # Save final model
    model.save('final_model.h5')
    
    return model, history

# Main execution
def main():
    # Prepare dataset paths and labels
    image_paths = []
    labels = []
    class_names = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
    
    for idx, class_name in enumerate(class_names):
        class_path = f'NEU-DET/{class_name}'
        for img_name in os.listdir(class_path):
            if img_name.endswith('.jpg'):
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(idx)
    
    # Split dataset
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = create_dataset(train_images, train_labels)
    val_dataset = create_dataset(val_images, val_labels, is_training=False)
    
    # Train model
    model, history = train_model(train_dataset, val_dataset)
    
    return model, history

if __name__ == "__main__":
    model, history = main()





"""
Key features of this implementation:

CPU Optimization:


Uses all 12 threads of your i7 processor
Optimized data pipeline with prefetching
Multi-core processing for data loading


ResNet50 Architecture:


Uses transfer learning with pre-trained ResNet50
Freezes early layers to speed up training
Added custom classification layers on top


Model Saving:


Saves best model during training ('best_model.h5')
Saves final model after training ('final_model.h5')
Implements early stopping to prevent overfitting


Memory Management:


Batch size of 32 (adjust if memory issues occur)
Efficient data loading with tf.data API
Image preprocessing optimized for ResNet50
"""