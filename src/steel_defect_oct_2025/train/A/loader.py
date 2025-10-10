import tensorflow as tf
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# For fine-tuning we need TFRecordDataset
train_dataset = tf.data.TFRecordDataset("train.tfrecord")
val_dataset   = tf.data.TFRecordDataset("val.tfrecord")

# TODO: parse TFRecord into images + labels here
