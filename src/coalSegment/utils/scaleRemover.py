import os
from matplotlib import pyplot as plt
import cv2 

folder_path = r"D:\NML ML Works\Coal photomicrographs\Coal_Lebels"

# files = os.listdir(folder_path)
files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
# for file in files:
#     imagine_dragen = plt.imread(os.path.join(folder_path, file))
#     print(file, " ", imagine_dragen.shape)

image = cv2.imread(os.path.join(folder_path, 'image 009.jpg'))
# Define rectangle coordinates (top-left and bottom-right corners)
top_left = (2144, 36)
bottom_right = (2572, 162)

# Draw filled rectangle on the image
cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)  # Red color with filled rectangle

cv2.imwrite(os.path.join(folder_path, 'scaleRemoved/cool.jpg'), image)
plt.imshow(image)
plt.axis("off")
plt.show()

# 2141, 28
# 2572, 161