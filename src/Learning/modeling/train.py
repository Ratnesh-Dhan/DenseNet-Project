import os 
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
from obj_detection_model import object_detection_model
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

IMG_SIZE = 224

def parse_annotatoin(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    obj = root.find('object')
    label = 1 if obj.find('name').text == 'dog' else 0

    bndbox = obj.find('bndbox')
    xmin = float(bndbox.find('xmin').text)
    ymin = float(bndbox.find('ymin').text)
    xmax = float(bndbox.find('xmax').text)
    ymax = float(bndbox.find('ymax').text)

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    # Normalize coordinates
    bbox = [xmin / width, ymin / height, xmax / width, ymax / height]
    return label, bbox

def load_dataset(img_dir):
    x, y_bbox, y_class = [], [], []
    for file in os.listdir(img_dir):
        if file.endswith(".jpg"):
            img_path = os.path.join(img_dir, file)
            xml_path = img_path.rsplit(".", 1)[0] + ".xml"

            if not os.path.exists(xml_path):
                continue

            # load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

            label, bbox = parse_annotatoin(xml_path)

            x.append(img)
            y_class.append(label)
            y_bbox.append(bbox)
    return np.array(x, dtype=np.float32), np.array(y_bbox, dtype=np.float32), np.array(y_class, dtype=np.float32)

x_train, y_bbox_train, y_class_train = load_dataset(r"C:\Users\NDT Lab\Software\DenseNet-Project\DenseNet-Project\Datasets\Asirra_cat_vs_dogs\train")
x_val, y_bbox_val, y_class_val = load_dataset(r"C:\Users\NDT Lab\Software\DenseNet-Project\DenseNet-Project\Datasets\Asirra_cat_vs_dogs\validation")

model = object_detection_model()
early_stop = EarlyStopping(
    monitor='val_loss',     # monitor total validation loss
    patience=10,            # stop after 10 epochs of no improvement
    restore_best_weights=True
)
history = model.fit(
    x_train,
    {
        "bbox": y_bbox_train, "class": y_class_train
    },
    validation_data=(x_val, {"bbox": y_bbox_val, "class": y_class_val}),
    epochs=100,
    batch_size=16,
    # callbacks=[early_stop]
)

model.save("object_detection_model.keras")
print("Done")

# --- Plot training history ---
plt.figure(figsize=(12,5))

# 1️⃣ Total loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 2️⃣ Bounding box loss
plt.subplot(1, 3, 2)
plt.plot(history.history['bbox_loss'], label='Train BBox Loss')
plt.plot(history.history['val_bbox_loss'], label='Val BBox Loss')
plt.title('BBox Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 3️⃣ Class loss
plt.subplot(1, 3, 3)
plt.plot(history.history['class_loss'], label='Train Class Loss')
plt.plot(history.history['val_class_loss'], label='Val Class Loss')
plt.title('Class Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()