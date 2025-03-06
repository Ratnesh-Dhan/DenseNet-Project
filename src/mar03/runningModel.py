import tensorflow as tf
import numpy as np
from PIL import Image
import cv2, matplotlib.pyplot as plt

image_path="./test2.jpg"
IMG_SIZE = (256, 256)  # Resize images to this size 
original_image = Image.open(image_path).convert("RGB")
resized_image = np.array(original_image.resize(IMG_SIZE)) / 255.0
model = tf.keras.models.load_model("mar_4_segmentation_model.h5")

prediction = model.predict(resized_image)[0]

print(prediction)