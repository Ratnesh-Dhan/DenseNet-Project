import os
from matplotlib import pyplot as plt
import cv2 

image_path = r"D:\NML ML Works\Coal photomicrographs\Coal_Lebels\image0003.jpg"
image = cv2.imread(image_path)

def floodFil(image, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right
    for i in range(y1,y2,1):
        pixel_value = image[i, x1-2]
        if all(pixel_value == 255):
            print(pixel_value)

top_left = (2144, 36)
bottom_right = (2572, 162)

# Draw filled rectangle on the image
cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)  # Red color with filled rectangle
floodFil(image, top_left, bottom_right)

plt.imshow(image)
plt.axis("off")
plt.show()