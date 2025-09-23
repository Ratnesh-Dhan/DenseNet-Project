import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
NUM_CLASSES = 6
backbone = keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(224,224,3),
    weights="imagenet"
)
backbone.trainable = False

inputs = keras.Inputs(shape=(224,224,3))
x = backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)

# Classificatoin branch
cls_output = layers.Dense(NUM_CLASSES, activation="sigmoid", name="cls_output")(x)

# Box regression branch (4 values: [xmin, ymin, xmax, ymax])
box_output = layers.Dense(4, activation="sigmoid", name="box_output")(x)

model = keras.Model(inputs, outputs=[cls_output, box_output])

losses = {
    "cls_output": "categorical_crossentropy",
    "box_output": "mse",  # or smooth L1 if you want custom
}

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=losses,
    metrics={"cls_output": "accuracy"}
)


# model = keras_cv.models.EfficientDet.from_preset(
#     "efficientdet_lite0",  # small + fast
#     num_classes=6,         # <-- your dataset classes
# )

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

model.save("./model/new.keras")