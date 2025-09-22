import tensorflow as tf
import keras_cv

# --- Dataset loader ---
def parse_tfrecord_fn(example):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    bboxes = tf.stack([
        tf.sparse.to_dense(example["image/object/bbox/ymin"]),
        tf.sparse.to_dense(example["image/object/bbox/xmin"]),
        tf.sparse.to_dense(example["image/object/bbox/ymax"]),
        tf.sparse.to_dense(example["image/object/bbox/xmax"]),
    ], axis=-1)

    labels = tf.sparse.to_dense(example["image/object/class/label"])

    target = {
        "boxes": bboxes,
        "classes": tf.cast(labels, tf.float32)
    }
    return image, target


def load_dataset(tfrecord_path, batch_size=8, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(512)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = load_dataset("train.tfrecord", batch_size=8)
val_ds   = load_dataset("val.tfrecord", batch_size=8, shuffle=False)

# --- Model ---
model = keras_cv.models.EfficientDet.from_preset(
    "efficientdet_lite0",  # small + fast
    num_classes=6,         # <-- your dataset classes
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    classification_loss="focal",
    box_loss="smoothl1",
)

# --- Training ---
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

model.save("new.keras")