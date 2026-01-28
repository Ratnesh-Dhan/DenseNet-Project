import cv2

image = cv2.imread("./images/image2.avif")
cv2.imwrite("./images/ingot.jpg", image)