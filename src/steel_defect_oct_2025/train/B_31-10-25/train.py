from tf_dataset_loader import load_dataset, preprocess, CLASS_NAMES
from model import get_model
import tensorflow as tf
import os

train_records = load_dataset(
    "Datasets/NEU-DET/train/annotations",
    "Datasets/NEU-DET/train/images"
)
val_records = load_dataset(
    "Datasets/NEU-DET/val/annotations",
    "Datasets/NEU-DET/val/images"
)

train_ds = tf.data.Dataset.from_generator(
    lambda: iter(train_records),
    output_types=(tf.string, tf.float32, tf.int32)
)
train_ds = train_ds.map(lambda x, y, z: preprocess(x, y, z)).batch(4).shuffle(100)

val_ds = tf.data.Dataset.from_generator(
    lambda: iter(val_records),
    output_types=(tf.string, tf.float32, tf.int32)
)
val_ds = val_ds.map(lambda x, y, z: preprocess(x, y, z)).batch(4)

model = get_model(CLASS_NAMES)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)
model.save("efficientdet_neu_det.keras")


