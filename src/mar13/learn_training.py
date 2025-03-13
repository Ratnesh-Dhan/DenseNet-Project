"""
https://www.geeksforgeeks.org/how-to-train-tensorflow-models-in-python/
https://www.learnpytorch.io/00_pytorch_fundamentals/
"""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset_name = 'cifar10'
(train_dataset, test_dataset), dataset_info = tfds.load(
    name=dataset_name,
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True
)

image, label = next(iter(train_dataset.take(1)))

num_classes = dataset_info.features['label'].num_classes
print(num_classes)

# Preprocessing & Normalization
def preprocessing_data(image, label):
    image = tf.cast(image,tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocessing_data)
test_dataset = test_dataset.map(preprocessing_data)
