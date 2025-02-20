from matplotlib import pyplot as plt
import cv2

image = plt.imread("color.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
edgey = cv2.Canny(gray, 20, 210)

plt.subplot(1,2,1)
plt.imshow(gray)
plt.axis("off")
plt.title("Gray")

plt.subplot(1,2,2)
plt.imshow(gray)
plt.axis("off")
plt.title("Canny")

plt.show()