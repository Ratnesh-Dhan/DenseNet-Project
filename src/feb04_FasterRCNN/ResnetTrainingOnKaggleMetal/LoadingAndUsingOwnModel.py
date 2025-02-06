# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:15:14 2025

@author: NDT Lab
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_and_predict(image_path):
    # Load saved model
    model = tf.keras.models.load_model('../../../MyTrained_Models/final_model.keras')
    
    # Preprocess single image
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    # Make prediction
    prediction = model.predict(image)
    class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence

# image_path = "../../img/"
# image_path = "../../img/Metal/testImage.bmp"
image_path = "../../img/Metal/Pincher.jpg"
image = plt.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predicted_class, confidence = load_and_predict(image_path)
print(f"Class: {predicted_class}, Confidence: {confidence}")
roundOff = round(float(confidence), 2)
plt.imshow(image)
plt.title(f"Class: {predicted_class}, Confidence: {roundOff}")
plt.axis("off")
plt.show()


