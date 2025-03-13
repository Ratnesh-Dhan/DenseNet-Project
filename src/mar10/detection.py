import tensorflow as tf


model = tf.keras.models.load_model("pcb_detector.h5")
import cv2, os, sys
import numpy as np, matplotlib.pyplot as plt

# Load and preprocess the image
image_path = "VID202106071443181-24_jpg.rf.a4d3bb6e402690a593c8b0da100c1509.jpg"
image_path = os.path.join("../../Datasets/pcbDataset/validation/img", image_path)
print(f"Image exists: {image_path} : ",os.path.exists(image_path))
image = cv2.imread(image_path)
image = cv2.resize(image, (416, 416))  # Resize to the input size of the model
image = image.astype('float32') / 255.0  # Normalize the image
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(image)
print(predictions)
