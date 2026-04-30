import tensorflow as tf
import os

def load_image(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (256, 256))
    mask = tf.cast(mask > 0, tf.float32)  # force binary

    return image, mask

def get_array_with_path(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    return files
