# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:59:46 2025
# WE ARE TRYING TO TRAIN RCNN MODEL WITH THE HELP OF PRE-TRAINED MODEL AND DATASET FROM KAGGLE

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
above is the the link where you can find the models . choose models according to your nature of task.

@author: NDT Lab
"""

import tensorflow as tf
import cv2


image_path = "../img/sample-street1.jpg"
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


# Convert tensor to numpy array and make a copy for drawing
display_image = image[0].numpy().copy()

# Get image dimensions
height, width = display_image.shape[:2]

# Draw boxes
for box, score, cls in zip(boxes[0], scores[0], classes[0]):
    if score > 0.5:
        y1, x1, y2, x2 = box
        
        # Convert normalized coordinates to pixel coordinates
        x1_pixel = int(x1 * width)
        y1_pixel = int(y1 * height)
        x2_pixel = int(x2 * width)
        y2_pixel = int(y2 * height)
        
        # Draw rectangle and text
        cv2.rectangle(display_image, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 255, 0), 2)
        label = f"Class {int(cls)}: {score:.2f}"
        cv2.putText(display_image, label, (x1_pixel, y1_pixel - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imshow("Prediction", display_image)
cv2.waitKey(0)