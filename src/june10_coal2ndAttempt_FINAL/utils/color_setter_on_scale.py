# this is to remove the scale by blackning the white font on the black background.
import cv2
import numpy as np
import os
import tqdm

# Path
sauce = r"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final\C1"
destination = r'D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\save\C1'
os.makedirs(destination, exist_ok=True)

# Load folder
imgs = os.listdir(sauce)

for image_name in tqdm.tqdm(imgs, desc="processing images"):
    # if(image_name.split('.')[-1].lower == "png" or image_name.split('.')[-1].lower == "jpg" or image_name.split('.')[-1].lower == "jpeg"):
    print(image_name)
    image_path = os.path.join(sauce, image_name)
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Clone for output
    result = img.copy()

    # Flood fill mask (2 pixels bigger than the image)
    h, w = gray.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Get exact white pixels only
    white_pixels = np.argwhere((img[:,:,0] == 255) & (img[:,:,1] == 255) & (img[:,:,2] == 255))

    for y, x in white_pixels:
        # Flood fill starting from this pure white pixel
        cv2.floodFill(result, mask, (x, y), (0, 0, 0), loDiff=(20,20,20), upDiff=(20,20,20))

    cv2.imwrite(os.path.join(destination, image_name), result)
