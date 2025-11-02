# https://keras.io/examples/vision/yolov8/
# https://datasetninja.com/traffic-vehicles-object-detection#class-balance
import os,cv2
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0

class_ids = [
    "car",
    "number_plate",
    "blur_number_plate",
    "tow_wheelers",
    "auto",
    "bus",
    "truck",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# Path to images and annotations
path_images = "../../../../Datasets/Traffic_Dataset/images/train/"
path_annot = "../../../../Datasets/Traffic_Dataset/labels/train/"

# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".txt")
    ]
)

# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
    ]
)

def parse_yolo_annotation(image_path,  path_annot):
    """
    Parses a YOLO-format annotation file and returns:
    - image_path
    - boxes (list of [xmin, ymin, xmax, ymax])
    - class_ids (list of ints)
    """
    # Derive corresponding image path
    txt_file = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    txt_path = os.path.join(path_annot, txt_file)

    # Read YOLO labels
    with open(txt_path, "r") as f:
        lines = f.readlines()

    boxes = []
    class_ids = []

    # You’ll need actual image size to convert from normalized → pixel coords
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w, _ = img.shape

    for line in lines:
        cls, x_c, y_c, bw, bh = map(float, line.strip().split())
        class_ids.append(int(cls))

        # Convert normalized YOLO → pixel coordinates
        x_c *= w
        y_c *= h
        bw *= w
        bh *= h

        xmin = x_c - bw / 2
        ymin = y_c - bh / 2
        xmax = x_c + bw / 2
        ymax = y_c + bh / 2

        boxes.append([xmin, ymin, xmax, ymax])

    return image_path, boxes, class_ids



image_paths = []
bbox = []
classes = []
for jpg_file in tqdm(jpg_files):
    image_path, boxes, class_ids = parse_yolo_annotation(jpg_file, path_annot)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)

print(image_paths[:10])
print(bbox[:10])
print(classes[:10])