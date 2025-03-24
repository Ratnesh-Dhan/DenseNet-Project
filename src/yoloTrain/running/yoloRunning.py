# yolo test 2

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model_name = "../../../MyTrained_Models/pcbYOLO/last.pt"

model = YOLO(model_name)

image = plt.imread("image4.jpg")
# if image.shape[0] < image.shape[1]:
#     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# results = model.track(image)

results = model(image)


for result in results:
    classes_names = result.names

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


# iterate over each box
for box in result.boxes:
    # check if confidence is greater than 40 percent
    if box.conf[0] > 0.4:
        # get coordinates
        [x1, y1, x2, y2] = box.xyxy[0]
        # convert to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # get the class
        cls = int(box.cls[0])

        # get the class name
        class_name = classes_names[cls]

        # class_name = "Cap" if class_name == "Mov" else class_name
        class_name = "IC" if class_name == "MOSFET" else class_name
        # print("class name: ", class_name)
        colour = getColours(cls)

        # draw the rectangle
        image = cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        
        print(class_name)
        # cv2.putText(image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        cv2.putText(image, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
   
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the image with detections
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()