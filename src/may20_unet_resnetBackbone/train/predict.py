# predict.py
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/unet_resnet50_corrosion.h5", compile=False)

img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (256, 256))
inp = np.expand_dims(img / 255.0, axis=0)

pred = model.predict(inp)[0, :, :, 0]
mask = (pred > 0.5).astype(np.uint8)

plt.subplot(1, 2, 1)
plt.imshow(img[..., ::-1])
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title("Predicted Mask")

plt.show()
