import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('/mnt/d/DATASETS/NEU-DET/images/train/crazing_18.jpg')
image = cv2.resize( image, (400,400))
# Define the sharpening kernel
# This kernel enhances the center pixel relative to its neighbors, highlighting edges.
sharpening_kernel = np.array([
    [ 0, -1, 0],
    [-1, 5, -1],
    [ 0, -1, 0],

])

# Apply the kernel to the image using filter2D
# The '-1' in the function specifies that the depth of the output image should be the same as the input
sharpend_image = cv2.filter2D(image, -1, sharpening_kernel)

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(sharpend_image)
plt.title("Sharpend Image")

plt.show()
plt.savefig("sharpen_crazing_image400*400.jpg")
# cv2.imshow('Original Image', image)
# cv2.imshow('Sharpened Image', sharpend_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save if needed
# cv2.imwrite('sharpened_output.jpg', sharpend_image)