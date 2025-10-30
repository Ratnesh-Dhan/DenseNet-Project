import glob
import tensorflow as tf
from dataset_loader import load_image_and_labels


train_xml_files = glob.glob("train/annotations/*.xml")
img_dir = "train/images"

def tf_load(xml_file):
    img, labels = tf.py_function(
        func=load_image_and_labels,
        inp=[xml_file, img_dir],
        Tout=[tf.float32, {"boxes": tf.float32, "labels": tf.int32}]
    )
    img.set_shape([200,200,3])
    labels["boxes"].set_shape([None,4])
    labels["labels"].set_shape([None])
    return img, labels

train_dataset = tf.data.Dataset.from_tensor_slices(train_xml_files)
train_dataset = train_dataset.map(tf_load).batch(8).prefetch(tf.data.AUTOTUNE)
