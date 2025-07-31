import cv2, numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image = cv2.imread("img/ingot2.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (3,3), 0) # For noise reduction

# Option 1
sharpen_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
image_kernel = cv2.filter2D(image, -1, sharpen_kernel) # Edge enhancement

# Option 2
adaptive_thresh = cv2.adaptiveThreshold(image, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2) # Block size 11, constant C = 2 (adjust these)

# Option 3
# The difference between the dilation and erosion of an image.
# Highlights the borders of objects.
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(image, kernel, iterations=1)
eroded = cv2.erode(image, kernel, iterations=1)
morph_gradient = cv2.subtract(dilated, eroded)

print(image.shape)
# plt.imshow(image, cmap="gray")
# plt.axis("off")
# plt.show()

plt.subplot(1,3,1)
plt.imshow(image_kernel, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(adaptive_thresh, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(morph_gradient, cmap="gray")
plt.axis("off")

plt.show()