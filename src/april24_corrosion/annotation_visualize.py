import json
import cv2
import matplotlib.pyplot as plt

location = r'C:\Users\NDT Lab\Pictures\dataset\roboflow-corrosion\valid\_annotations.coco.json'
with open(location) as f:
    data = json.load(f)
# Load all images and annotation
count = 0
while count < len(data['images']):
    image_info = data['images'][count]
    image_path = f"C:/Users/NDT Lab/Pictures/dataset/roboflow-corrosion/valid/{image_info['file_name']}"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get annotations for this image
    anns = [ann for ann in data['annotations'] if ann['image_id'] == image_info['id']]

    # Draw bounding boxes
    for ann in anns:
        x, y, w, h = ann['bbox']
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        print(f"Image size: {img.shape}")
        print(f"BBox: {ann['bbox']}")


    plt.imshow(img)
    plt.axis('off')
    plt.show()
    count = count + 1
