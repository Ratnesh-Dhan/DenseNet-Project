import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image, save=False):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    if save:
        plt.savefig(f"{title}.png")
    plt.show()

# Load image
image_path = "this.jpeg"
image = cv2.imread(image_path)

# Resize if needed
image_resized = cv2.resize(image, (512, 512))  # Optional
image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# CLAHE - try different clipLimit and tileGridSize
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
clahe_img = clahe.apply(image_gray)

# Thresholding to segment
_, threshold_img = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display results
display_image("CLAHE Enhanced Image (size 512x512)", clahe_img)
# display_image("Segmented (Thresholded) Image", threshold_img)
