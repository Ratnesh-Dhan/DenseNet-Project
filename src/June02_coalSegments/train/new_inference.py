import numpy as np
import cv2, os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def sliding_window_inference(model, image, support_image, window_size=31, stride=10, center_patch_size=5):
    h, w, _ = image.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    half_patch = window_size // 2
    half_center = center_patch_size // 2

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image[y:y+window_size, x:x+window_size]
            support_patch = support_image[y:y+window_size, x:x+window_size]
            if np.all(support_patch == 0):
                print(f"Skipping patch at {x}, {y} because it's all zeros")
                continue
            patch = patch.astype(np.float32) / 255.0
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
image_name = "1.jpg"
input_image = Image.open(f"../img/{image_name}")
support_image = Image.open(f"../support_img/{image_name}")
heatmap = sliding_window_inference(model, input_image, support_image)
cv2.imwrite(f"{image_name}", heatmap)
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(heatmap)
plt.title("Segmentation Result")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{image_name}_comparison.png")
plt.show()
