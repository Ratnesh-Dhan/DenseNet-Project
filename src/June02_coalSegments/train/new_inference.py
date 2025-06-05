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
            if pred_class == 0: 
                color = (0, 255, 0) #organic
            elif pred_class == 1:
                color = (0, 0, 255) #inorganic
            else:
                color = (255, 0, 0) #background ( supposed to be but not in the model . Mostly it will be black)

            cv2.rectangle(
                heatmap,
                (center_x - half_center, center_y - half_center),
                (center_x + half_center, center_y + half_center),
                color,
                thickness=-1
            )
    return heatmap

model = tf.keras.models.load_model("../models/working_best_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
image_name = "image 1"
input_image = plt.imread(f"./Images/{image_name}.jpg")
support_image = Image.open(f"./supportImages/{image_name}.png").convert("RGB")
support_image = np.array(support_image).astype(np.uint8)
heatmap = sliding_window_inference(model, input_image, support_image)
cv2.imwrite(f"./results/{image_name}_heatmap.png", heatmap)
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(heatmap)
plt.title("Segmentation Result")
plt.axis('off')
plt.tight_layout()
plt.savefig(f"./results/{image_name}_comparison.png")
plt.show()
