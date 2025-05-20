# dataset_loader.py
import cv2
import numpy as np
import os
import tensorflow as tf

class CorrosionDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=8, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_ids = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_ids) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        images = []
        masks = []
        for id in batch_ids:
            img_path = os.path.join(self.image_dir, id)
            mask_path = os.path.join(self.mask_dir, id.replace('.jpg', '.png'))

            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img / 255.0

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)
