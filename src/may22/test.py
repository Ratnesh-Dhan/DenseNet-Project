import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
def display_image(title, image, save=True):
    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    if save:
        plt.savefig(f"{title}.png")
    plt.show()

image_path = "this.jpeg"
image = cv2.imread(image_path)
# image_resized = cv2.resize(image, (500, 600))
image_resized = image
image_bw = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=5)
clahe_img = np.clip(clahe.apply(image_bw) + 30, 0, 255).astype(np.uint8)
_, threshold_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
display_image("Ordinary Threshold", threshold_img, save=False)
display_image("CLAHE Image", clahe_img, save=True)