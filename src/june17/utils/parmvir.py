from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

image = Image.open("my.jpeg")
image = np.array(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image[image > 90] = 255
print(image.shape)
# (480, 640, 3)

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
left = -1
right = -1

for i in range(0, 480):
    for j in range(0, 640):
        if image[i][j] < 250:
            left = i
            print(left, " : ", image[i][j])
            break
