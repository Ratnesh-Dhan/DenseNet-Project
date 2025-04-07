# yolo test 2
import random
import colorsys
import cv2
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

model_name = "../../../MyTrained_Models/pcb/best_7_april.pt"

model = YOLO(model_name)

image = plt.imread("../resis.jpg")
# image = cv2.resize(image, (640, 480))
print(f"image shape : {image.shape}")
# if image.shape[0] < image.shape[1]:
#     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# results = model.track(image)

results = model(image)


for result in results:
    classes_names = result.names

def getColours(seed=None):
    if seed is not None:
        random.seed(seed + 6)
    
    # Generate a random hue
    hue = random.random()
    
    # Set saturation and value (brightness) to create a darker color
    saturation = 0.7 + random.random() * 0.3  # 70-100% saturation
    value = 0.4 + random.random() * 0.2  # 40-60% brightness
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    
    # Convert to 0-255 range and create tuple
    rgb_tuple = tuple(int(x * 255) for x in rgb)
    
    return rgb_tuple

# iterate over each box
for box in result.boxes:
    # check if confidence is greater than 40 percent
    # if box.conf[0] > 0.4:
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
        print(colour)

        # draw the rectangle
        image = cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        
        print(class_name)
        cv2.putText(image, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        # cv2.putText(image, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
   
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the image with detections
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()