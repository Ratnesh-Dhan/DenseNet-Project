import tensorflow as tf
import matplotlib.pyplot as plt

tfrecord_path = "/mnt/z/DATASETS/corrosion_tfr/train/corrosion.tfrecord"

def parse_example(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.image.decode_png(example["image"], channels=3)
    mask = tf.image.decode_png(example["mask"], channels=1)

    return image, mask

dataset = tf.data.TFRecordDataset(tfrecord_path)
dataset = dataset.map(parse_example)

# visualize few samples
for img, mask in dataset.take(5):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title("Image")

    plt.subplot(1,2,2)
    plt.imshow(mask.numpy().squeeze(), cmap="gray")
    plt.title("Mask")

    plt.show()