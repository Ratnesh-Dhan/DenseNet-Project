# -*- coding: utf-8 -*-
# USING EXISTING MODEL FROM KAGGLE NAMED deeplabv3 FOR SEGMENTAION

import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

model_path = "../../TrainedModel/2.tflite"

#We cant use this syntax because "load_model" wont work with "tflite" models. but will work with .keras
# model = tf.keras.models.load_model(model_path)

try:
    #So we will use different syntax to load .tflite model
    #Load tflite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    #Get input and output tensor ( this is only for .tfmodel )
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def preprocess_image(image_path):
        img = cv2.imread(image_path)
        # img = cv2.resize(img, (512,512))
        img = cv2.resize(img, (257,257)) # modified according to .tflite model
        img = img/255.0
        img = img.astype(np.float32) #Ensure the image is Float32 ( only for .tflite model )
        return np.expand_dims(img, axis=0)

    def visualization_segmentation(image_path, mask):
        og_image = cv2.imread(image_path)
        og_image = cv2.resize(og_image, (512,512))
        plt.subplot(1,2,1)
        plt.title("Og image")
        plt.imshow(og_image)
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Segmented image")
        plt.imshow(mask, cmap='jet', alpha=0.5)
        plt.axis("off")

        plt.show()

    image_path ="../img/scenery.jpg"
    processed_image = preprocess_image(image_path)

    # NOW WE ARE DOWN TO RUNNING THE MODEL TO SEGMENT
    # for .tflite model
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke() # running the model
    predictions = interpreter.get_tensor(output_details[0]['index'])
    # / for .tflite model
    # predictions = model.predict(processed_image) ##This line will be used when the model has not .tflite file extension   
    mask = np.argmax(predictions, axis=-1) #Get the predicted class for each pixel
    mask = np.squeeze(mask) #Remove batch dimension

    visualization_segmentation(image_path, mask)
except Exception as e:
    print(f"Error : {e}")