# predict.py
import matplotlib.pyplot as plt
import cv2, os
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../models/unet_resnet50_corrosion_office_version.keras", compile=False)

this_path = "../../may15_corrosion_latest/train/test_images/"
images = os.listdir(this_path)
for i in images:
    if not i.endswith(".jpg"):
        continue
    img = cv2.imread(os.path.join(this_path, i))
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
