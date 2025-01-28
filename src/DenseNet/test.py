import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained DeepLabV3 model from TensorFlow Hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/segmentation/deeplabv3/1", input_shape=(None, None, 3))
])

# Function to preprocess the input image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (512, 512))  # Resize to model input size
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to visualize the results
def visualize_segmentation(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmentation Mask")
    plt.imshow(mask)
    plt.axis("off")
    plt.show()

# Load and preprocess an image
image_path = '../img/cat1.jpg'  # Replace with your image path
image = preprocess_image(image_path)

# Add batch dimension
input_image = tf.expand_dims(image, axis=0)

# Run the model to get the segmentation mask
predictions = model(input_image)
mask = tf.argmax(predictions, axis=-1)
mask = tf.squeeze(mask)  # Remove batch dimension

# Visualize the results
visualize_segmentation(image, mask.numpy())
