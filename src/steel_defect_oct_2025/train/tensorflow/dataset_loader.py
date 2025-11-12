import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf

class XMLDatasetTF(tf.keras.utils.Sequence):
    def __init__(self, split, root_dir, classes_file, batch_size=8, img_size=512):
        self.img_dir = os.path.join(root_dir, "images", split)
        self.ann_dir = os.path.join(root_dir, "annotations", split)
        self.img_size = img_size
        self.batch_size = batch_size

        with open(classes_file) as f:
            self.classes = [line.strip() for line in f]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.imgs = [f for f in os.listdir(self.img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return int(np.ceil(len(self.imgs) / self.batch_size))

    def __getitem__(self, idx):
        batch_imgs = self.imgs[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for name in batch_imgs:
            img_path = os.path.join(self.img_dir, name)
            xml_path = os.path.join(self.ann_dir, os.path.splitext(name)[0] + ".xml")

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype("float32") / 255.0

            label_vec = np.zeros(len(self.classes), dtype="float32")

            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                for obj in tree.findall("object"):
                    cname = obj.find("name").text
                    if cname in self.class_to_idx:
                        label_vec[self.class_to_idx[cname]] = 1.0

            images.append(img)
            labels.append(label_vec)

        return np.stack(images), np.stack(labels)
