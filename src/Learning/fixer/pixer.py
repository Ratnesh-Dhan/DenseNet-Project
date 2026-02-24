import cv2

image_path = '/home/zumbie/Documents/form_documents/original.jpg'
image = cv2.imread(image_path)

image = cv2.resize(image, (140, 60)) 

cv2.imwrite('/home/zumbie/Documents/form_documents/signature_fixed.jpg', image)