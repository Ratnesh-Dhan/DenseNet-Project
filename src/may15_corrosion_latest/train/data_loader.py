import tensorflow as tf
import os

# IMG_SIZE = (512, 512)
IMG_SIZE = (256, 256)

# ---------------------------
# Load image + mask
# ---------------------------
def load_image_mask(image_path, mask_path):
    # -------- IMAGE --------
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    # -------- MASK --------
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, IMG_SIZE, method='nearest')

    # Simple rule: non-zero = corrosion
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.cast(mask > 0, tf.float32)

    return image, mask


# ---------------------------
# Augmentation (only for training)
# ---------------------------
def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask


# ---------------------------
# Dataset builder
# ---------------------------
def get_dataset(image_dir, mask_dir, batch_size=8, training=True):
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    mask_paths = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    # if training:
    #     dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # ⚠️ If RAM is limited, use: dataset.cache("/tmp/cache")
    dataset = dataset.cache()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset