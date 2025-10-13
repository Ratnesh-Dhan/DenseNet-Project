import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random, sys

# === Paths ===
images_dir = "../../../Datasets/NEU-DET/train/images"
annotations_dir = "../../../Datasets/NEU-DET/train/annotations"
base_dir = "../../../Datasets/NEU-DET/train"
# === Helper to Parse VOC XML ===
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    return boxes, labels

# === Visualization Function ===
def visualize_sample(image_path, annotation_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes, labels = parse_voc_annotation(annotation_path)

    for (box, label) in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, label, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# === Collect all images with full paths ===
all_images = []
for cls in os.listdir(images_dir):
    cls_folder = os.path.join(images_dir, cls)
    if os.path.isdir(cls_folder):
        for img_file in os.listdir(cls_folder):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                all_images.append(os.path.join(cls_folder, img_file))

# === Show random samples ===
sample_images = random.sample(all_images, 5)

for img_path in sample_images:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(annotations_dir, filename + ".xml")
    if os.path.exists(ann_path):
        visualize_sample(img_path, ann_path)
    else:
        print(f"Annotation missing for {filename}")
sys.exit(0)
# === Helper to Parse Annotations ===
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels

# === Visualization Function ===
def visualize_sample(image_path, annotation_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes, labels = parse_voc_annotation(annotation_path)

    for (box, label) in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        color = (255, 0, 0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, label, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def parse_voc_annotation_1(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    return filename

def load_image_path(xml_file, img_dir):
    filename = parse_voc_annotation_1(xml_file)
    crop_name = filename.split("_")[0]
    if crop_name == "pitted":
        insider_folder_name = "pitted_surface"
    elif crop_name == "rolled-in":
        insider_folder_name = "rolled-in_scale"
    else:
        insider_folder_name = crop_name
    print("filename : ", filename)
    if filename.endswith(".jpg"):
        return os.path.join(img_dir, insider_folder_name, filename)
    else:
        print(filename)
        return os.path.join(img_dir, insider_folder_name, f'{filename}.jpg')

# === Pick a random file to test ===
# annotations = os.listdir(annotations_dir)
# random.shuffle(annotations)
# for i in annotations:
#     ann_path = os.path.join(annotations_dir, i)
#     print("ANN PATH ",ann_path)
#     img_path = load_image_path(i, images_dir)
#     visualize_sample(img_path, ann_path)

proper_image_path = os.path.join(images_dir, "crazing")
images = os.listdir(proper_image_path)
for i in images:
    xml_filename = i.split('.')[0] + ".xml"
    image_path = os.path.join(proper_image_path, i)
    ann_path = os.path.join(annotations_dir, xml_filename)
    visualize_sample(image_path, ann_path)
