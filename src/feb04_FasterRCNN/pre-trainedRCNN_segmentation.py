# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:21:35 2025

http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz
above link - model used in this program to perform image segmentation

@author: NDT Lab
"""
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "../img/sample-street1.jpg"
model_dir = "../../pre-Trained_model/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/saved_model"
model = tf.saved_model.load(model_dir)
infer = model.signatures['serving_default']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (1024, 1024))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.uint8)
    return image

image = preprocess_image(image_path)
outputs = infer(image)

detection_boxes = outputs['detection_boxes'].numpy()
detection_masks = outputs['detection_masks'].numpy()
detection_scores = outputs['detection_scores'].numpy()
detection_classes = outputs['detection_classes'].numpy()

# converting tensor to numpy array
display_image = image[0].numpy().copy()
height, width = display_image.shape[:2]

# creating a mask image
mask_image = np.zeros((height, width, 3), dtype=np.uint8)

#processing each detection
for i in range(len(detection_scores[0])):
    score = detection_scores[0][i]
    if score > 0.5:  # Confidence threshold
        mask = detection_masks[0][i]
        box = detection_boxes[0][i]
        
        # Convert normalized box coordinates to pixel coordinates
        y1, x1, y2, x2 = box
        x1 = int(x1 * width)
        x2 = int(x2 * width)
        y1 = int(y1 * height)
        y2 = int(y2 * height)
        
        #Resize mask to box size
        box_height = y2 - y1
        box_width = x2-x1
        mask = cv2.resize(mask, (box_width, box_height))
        
        color = np.random.randint(0,255,3, dtype=np.uint8)
        
        #Apply mask
        mask_region = np.zeros((height, width, 3), dtype=np.uint8)
        mask_region[y1:y2, x1:x2] = color * mask[..., np.newaxis]
        
        #Blend with original image
        alpha = 0.5 #Transparency factor
        mask_image = cv2.addWeighted(mask_image, 1, mask_region, alpha, 0)
        
        #Draw bounding box
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color.tolist(), 2)
        
        # Add label
        label = f"Class {int(detection_classes[0][i])}: {score:.2f}"
        cv2.putText(display_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
       
# Combine original image with mask
final_image = cv2.addWeighted(display_image, 1, mask_image, 0.5, 0)

# # Display results
# plt.figure(figsize=(12, 8))
# plt.imshow(final_image)
# plt.axis('off')
# plt.show()

# OpenCV display
final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
cv2.imshow("Instance Segmentation", final_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()