import cv2
import os 

image_path = "/home/zumbie/Documents/Form_Documents/MY-Signature.jpg"

image = cv2.imread(image_path)
h, w, c = image.shape

image = cv2.resize(image, (int(w/1.56), int(h/1.56)))
cv2.imwrite(f"{os.path.dirname(image_path)}/resized_image.jpg", image)
