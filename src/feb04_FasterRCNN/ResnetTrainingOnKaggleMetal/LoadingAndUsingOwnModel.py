# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:15:14 2025

@author: NDT Lab
"""
import tensorflow as tf
import numpy as np

def load_and_predict(image_path):
    # Load saved model
    model = tf.keras.models.load_model('best_model.h5')
    
    # Preprocess single image
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    # Make prediction
    prediction = model.predict(image)
    class_names = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    return predicted_class, confidence