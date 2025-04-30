import tensorflow as tf
import os
import random
from tensorflow.keras.applications.resnet50 import preprocess_input

class DataPipeline:
    def __init__(self, image_dir, corrosion_mask_dir, piece_mask_dir ):
        # Paths
        self.IMAGE_DIR = image_dir
        self.CORROSION_MASK_DIR = corrosion_mask_dir
        self.PIECE_MASK_DIR = piece_mask_dir

    def path_setter(self, image_dir, corrosion_mask_dir, piece_mask_dir):
        self.IMAGE_DIR = image_dir
        self.CORROSION_MASK_DIR = corrosion_mask_dir
        self.PIECE_MASK_DIR = piece_mask_dir

    def load_image_and_masks(self, image_path):
        # Derive corresponding mask paths
        filename = tf.strings.split(image_path, os.sep)[-1]
        corrosion_mask_path = tf.strings.join([self.CORROSION_MASK_DIR, filename], separator=os.sep)
        piece_mask_path = tf.strings.join([self.PIECE_MASK_DIR, filename], separator=os.sep)

        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        # image = tf.image.resize(image, image_size)
        image = tf.image.resize_with_pad(image, target_height=512, target_width=512)
        # image = tf.cast(image, tf.float32) / 255.0
        image = preprocess_input(tf.cast(image, tf.float32))

        # Load sample piece mask
        piece_mask = tf.io.read_file(piece_mask_path)
        piece_mask = tf.image.decode_png(piece_mask, channels=1)
        piece_mask = tf.image.resize_with_pad(piece_mask, 512, 512, method='nearest')
        piece_mask = tf.squeeze(piece_mask, axis=-1)

        # Load corrosion mask
        corrosion_mask = tf.io.read_file(corrosion_mask_path)
        corrosion_mask = tf.image.decode_png(corrosion_mask, channels=1)
        corrosion_mask = tf.image.resize_with_pad(corrosion_mask, 512, 512, method='nearest')
        corrosion_mask = tf.squeeze(corrosion_mask, axis=-1)

        # Create a final mask
        # final_mask = tf.zeros_like(piece_mask, dtype=tf.uint8)
        final_mask = tf.zeros_like(piece_mask, dtype=tf.float32)

        # final_mask = tf.where(piece_mask > 0, 1, final_mask)        # sample piece => class 1
        # final_mask = tf.where(corrosion_mask > 0, 2, final_mask)     # corrosion => class 2
        final_mask = tf.where(piece_mask > 0, tf.constant(1.0, dtype=final_mask.dtype), final_mask)        # sample piece => class 1
        final_mask = tf.where(corrosion_mask > 0, tf.constant(2.0, dtype=final_mask.dtype), final_mask)     # corrosion => class 2

        # One-hot encode final mask
        # final_mask = tf.one_hot(final_mask, depth=3, dtype=tf.float32)
        final_mask = tf.one_hot(tf.cast(final_mask, tf.int32), depth=3, dtype=tf.float32)

        return image, final_mask

    # --- Augmentation ---
    def augment(self, image, mask):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

        if tf.random.uniform(()) > 0.5:
            k = random.randint(1, 3)  # rotate 90, 180, 270
            image = tf.image.rot90(image, k=k)
            mask = tf.image.rot90(mask, k=k)

        return image, mask

    # --- Dataset Final Function ---
    def get_dataset(self, image_dir, batch_size=8, augment_data=False, shuffle=True):
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(lambda img: self.load_image_and_masks(img),
                            num_parallel_calls=tf.data.AUTOTUNE)

        # if augment_data:
        #     dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(100)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
