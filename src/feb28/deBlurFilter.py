import cv2
import numpy as np
import matplotlib.pyplot as plt

# image = cv2.imread('image-motion-blur.jpg')
image = cv2.imread('../img/image-motion-blur.jpg')
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(image, -1, sharpen_kernel)

# cv2.imshow('sharpen', sharpen)
# cv2.waitKey()

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharpen)
plt.title("clear")
plt.axis("off")

plt.show()