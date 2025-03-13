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

plt.imshow(image)
plt.title(label.numpy())
plt.axis("off")
plt.show()