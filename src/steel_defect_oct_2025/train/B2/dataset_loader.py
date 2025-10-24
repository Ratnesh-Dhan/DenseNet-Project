import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np

CLASS_MAP = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}

def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)

    boxes, labels = [], []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text) / width
        ymin = int(bndbox.find("ymin").text) / height
        xmax = int(bndbox.find("xmax").text) / width
        ymax = int(bndbox.find("ymax").text) / height
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_MAP[label])

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32)

def load_image_and_labels(xml_file, img_dir, img_size=(200,200)):
    filename = os.path.splitext(os.path.basename(xml_file))[0]
    
    # Find which folder contains the image
    for folder in os.listdir(img_dir):
        img_path = os.path.join(img_dir, folder, filename + ".jpg")
        if os.path.exists(img_path):
            break
    else:
        raise FileNotFoundError(f"Image {filename}.jpg not found in any class folder")

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size) / 255.0

    boxes, labels = parse_voc_annotation(xml_file)

    return img, {"boxes": boxes, "labels": labels}
