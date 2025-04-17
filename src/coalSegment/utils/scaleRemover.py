import os
from matplotlib import pyplot as plt
import cv2 

def scale_remover(image, top_left=(2123, 36), bottom_right=(2572, 162)):
    """
    Removes the scale from a coal photomicrograph image by drawing a black rectangle over it.
    
    Args:
        image (numpy.ndarray): Input image
        top_left (tuple): Top-left coordinates of the scale (default: (2123, 36))
        bottom_right (tuple): Bottom-right coordinates of the scale (default: (2572, 162))
    
    Returns:
        numpy.ndarray: Image with scale removed
    """
    # Draw filled rectangle on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    return image

if __name__ == '__main__':
    # Example usage
    folder_path = r"D:\NML ML Works\Coal photomicrographs"
    files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]

    for file in files:
        image = cv2.imread(os.path.join(folder_path, file))
        image = scale_remover(image=image)
        cv2.imwrite(os.path.join(folder_path, 'scaleRemoved/'+file), image)


# 2141, 28
# 2572, 161