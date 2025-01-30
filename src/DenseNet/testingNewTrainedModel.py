# -*- coding: utf-8 -*-
# WE ARE TESTING THE NEW TRAINED MODEL FOR CLASSIFICATION OF OBJECTS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

#loading saved model
try:
    model = tf.keras.models.load_model('../../MyTrained_Models/densenet_cifar10.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
img_path = "../img/cat1.jpg"
img = image.load_img(img_path, target_size=(32,32)) #Resizing to cifar-10 size
img_array = image.img_to_array(img)
img_array = img_array.astype('float32')/255
img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

#Making predictions
prediction = model.predict(img_array)

print(prediction)