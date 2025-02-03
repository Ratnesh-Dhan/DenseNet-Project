# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:50:16 2025
#THIS PROGRAM IS WRITTEN BY MICROSOFT COPILOT ( this is for model testing and the code iteslf )
@author: NDT Lab
"""

import tensorflow as tf
import cv2
import numpy as np

from TrainingDenseNet import DenseLayer, DenseBlock, TransitionLayer, DenseNet

with tf.keras.utils.custom_object_scope({'DenseLayer': DenseLayer, 'DenseBlock': DenseBlock, 'TransitionLayer': TransitionLayer, 'DenseNet': DenseNet}):
    model = tf.keras.models.load_model('../../TrainedModel/densenet_cifar10.h5')

img_path = "../img/cat1.jpg"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))  # Resize to 32x32 pixels
    image = image.astype('float32') / 255  # Normalize the pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

preprocessed_image = preprocess_image(img_path)
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class: ", predicted_class)

class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

print("Predicted class: ", predicted_class)
print("Predicted label: ", class_names[predicted_class[0]])
