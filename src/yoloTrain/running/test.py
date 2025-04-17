import cv2
import matplotlib.pyplot as plt

image_path = "../50.webp"

image = cv2.imread(image_path)
image = cv2.resize(image, (640,640))
image = cv2.rectangle(image, (170, 168) ,(414, 397), (0,0,0),2)
image = cv2.rectangle(image, (240, 225), (462, 456), (0,255,0),2)

plt.imshow(image)
plt.axis("off")
plt.show()