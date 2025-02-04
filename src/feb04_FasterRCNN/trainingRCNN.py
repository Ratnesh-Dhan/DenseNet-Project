# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:59:46 2025
# WE ARE TRYING TO TRAIN RCNN MODEL WITH THE HELP OF PRE-TRAINED MODEL AND DATASET FROM KAGGLE
@author: NDT Lab
"""

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
# import numpy as np

image_path = "../img/sample-street.jpg"
model_dir = '../../pre-Trained_model/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_dir)
if model:
    print("we have model")
else:
    print("we dont have model")
infer = model.signatures['serving_default']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (640,640)) #size image according to the model
    image = tf.expand_dims(image, axis=0) # Add batch dimension
    image = tf.cast(image, dtype=tf.uint8) #Convert to unit8
    return image

image = preprocess_image(image_path)

#Running inference
outputs =infer(image)

#Extract result
boxes = outputs['detection_boxes'].numpy()
scores = outputs['detection_scores'].numpy()
classes = outputs['detection_classes'].numpy()

#Visualize results
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for box, score, cls in zip(boxes[0], scores[0], classes[0]):
    if score > 0.5:
        print(box , score)
        y1, x1, y2, x2 = box
        print("this is y1 and x1 coordinates of the box ",y1," ", x1)
        # x1, y1, x2, y2 = box
        y1, x1, y2, x2 = int(y1 * image.shape[1]), int(x1 * image.shape[2]), int(y2 * image.shape[1]), int(x2 * image.shape[2])
        cv2.rectangle(image[0].numpy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
        # image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image[0].numpy(), f"Class {int(cls)}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,0,0), 2)
        # cv2.putText(image, f"Class {int(cls)}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,0,0), 2)
    
# plt.imshow(image[0].numpy())
# plt.axis('off')
# plt.show()
cv2.imshow("Prediction", image[0].numpy())
cv2.waitKey(0)
        