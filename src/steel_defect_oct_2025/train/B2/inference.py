import numpy as np
import tensorflow as tf
from dataset_loader import load_image_and_labels

def run_inference_ssd(model, image_path, max_boxes=10, threshold=0.3):
    img, _ = load_image_and_labels(image_path, "train/images")
    input_img = tf.expand_dims(img, axis=0)

    bbox_preds, class_preds = model.predict(input_img)
    bbox_preds = np.squeeze(bbox_preds)   # shape: (max_boxes,4)
    class_preds = np.squeeze(class_preds) # shape: (max_boxes,num_classes)

    boxes, labels, scores = [], [], []

    for i in range(max_boxes):
        score = np.max(class_preds[i])
        if score < threshold:
            continue
        label = np.argmax(class_preds[i])
        boxes.append(bbox_preds[i])
        labels.append(label)
        scores.append(score)

    return boxes, labels, scores
