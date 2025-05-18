import tensorflow as tf
import os

IMG_SIZE = (256, 256)

def load_image_mask(image_path, mask_path):
    # Load and preprocess image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    # Load and preprocess mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)  # decode RGB
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')

    # Convert red areas to 1, everything else to 0
    red_mask = tf.equal(mask, [255, 0, 0])
    red_mask = tf.reduce_all(red_mask, axis=-1)
    mask = tf.cast(red_mask, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)  # Add channel dimension

    return image, mask

def get_dataset(image_dir, mask_dir, batch_size=16):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset
