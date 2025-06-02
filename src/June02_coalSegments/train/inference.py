import numpy as np
import cv2, os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def sliding_window_inference(model, image, window_size=31, stride=10, center_patch_size=5):
    h, w, _f = image.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    half_patch = window_size // 2
    half_center = center_patch_size // 2

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[y:y+window_size, x:x+window_size].astype(np.float32) / 255.0
            # patch = np.expand_dims(patch, axis=(0, -1))  # shape: (1, 31, 31, 1)
            patch = np.expand_dims(patch, axis=0)  # for RGB 

            prediction = model.predict(patch, verbose=0)
            pred_class = np.argmax(prediction[0])

            center_y = y + half_patch
            center_x = x + half_patch
            color = (0, 255, 0) if pred_class == 0 else (0, 0, 255)  # Organic or Inorganic

            cv2.rectangle(
                heatmap,
                (center_x - half_center, center_y - half_center),
                (center_x + half_center, center_y + half_center),
                color,
                thickness=-1
            )
    return heatmap

files = os.listdir("../img")
model = tf.keras.models.load_model("model.keras")
for file in files:
    img = plt.imread(f"../img/{file}")
    heatmap = sliding_window_inference(model, img)
    plt.imshow(heatmap)
    plt.axis('off')
    plt.show()
