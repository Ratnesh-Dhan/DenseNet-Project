# dataset_loader.py
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

class CorrosionDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=16, image_size=256, **kwargs):
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_ids = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        images = []
        masks = []
        for id in batch_ids:
            img_path = os.path.join(self.image_dir, id)
            mask_path = os.path.join(self.mask_dir, id.replace('.jpg', '.png'))

            # Load and preprocess image
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, (self.image_size, self.image_size))
            # image = tf.cast(image, tf.float32) / 255.0
            image = tf.cast(image, tf.float32)  # Don't normalize here!
            image = preprocess_input(image)     # Apply ResNet preprocessing here

            # Load and preprocess mask (as RGB)
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=3)
            mask = tf.image.resize(mask, (self.image_size, self.image_size), method='nearest')

            # Select only pixels with RGB == [128, 0, 0]
            red_mask = tf.equal(mask, [128, 0, 0])
            red_mask = tf.reduce_all(red_mask, axis=-1)  # shape: (H, W)
            mask = tf.cast(red_mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)  # shape: (H, W, 1)

            images.append(image.numpy())
            masks.append(mask.numpy())

        return np.array(images), np.array(masks)
