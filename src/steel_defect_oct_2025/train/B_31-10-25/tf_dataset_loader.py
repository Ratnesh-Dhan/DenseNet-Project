import tensorflow as tf
import xml.etree.ElementTree as ET
import os

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

def parse_voc(xml_path, img_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_name = root.find("filename").text
    img_path = os.path.join(img_dir, img_name)

    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)

    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = int(bnd.find("xmin").text) / w
        ymin = int(bnd.find("ymin").text) / h
        xmax = int(bnd.find("xmax").text) / w
        ymax = int(bnd.find("ymax").text) / h
        boxes.append([ymin, xmin, ymax, xmax])
        labels.append(CLASS_MAP[name])

    return img_path, boxes, labels

def load_dataset(annotation_dir, img_dir):
    xml_files = [os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith(".xml")]
    records = [parse_voc(x, img_dir) for x in xml_files]
    return records

def preprocess(img_path, boxes, labels):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (512, 512))
    img = img / 255.0

    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    return {"images": img, "bounding_boxes": {"boxes": boxes, "classes": labels}}
