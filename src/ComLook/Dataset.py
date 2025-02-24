# COMLOOK 
from typing import List, Dict
import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
import json

class DatasetOperation:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.class_mapping = {
            # #TODO
        }
    
    def load_annotation(self, json_path: Path) -> List[Dict]:
        with open(json_path, 'r') as f:
            data = json.load(f)
        components = []
        # Below line needs to be modified according to my own dataset
        for obj in data['objects']:
            # Here we r trying to extract component information , again it needs to be tweeked acording to my own dataset
            component = {
                'class': obj['classTitle'],
                'bbox': obj['points']['exterior'],
                'image_size': (data['size']['width'], data['size']['height'])
            }
            components.append(component)
        return components
    
    def extract_component(self, image: np.ndarray, bbox: List[List[int]]) -> np.ndarray:
        # Get coordinates
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        # Extract component
        component = image[y1:y2, x1:x2]

        # Preprocess component
        component = self.preprocess_image(component)
        return component
       
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess component image
        """
        # Resize this to our model size . we have to change this and similarly in create_model we have to change the image shape . 
        image = cv2.resize(image, (224, 224))
        return image
 
    def create_data_generator(self, split: str, batch_size: int = 32):
        """
        Create data generator for specified split
        """
        split_path = self.base_path / split
        img_path = split_path / 'img'
        ann_path = split_path / 'ann'

        def generator():
            # Get all image files
            image_files = list(img_path.glob('*.jpg'))
            np.random.shuffle(image_files)

            batch_images = []
            batch_labels = []

            for img_file in image_files:
                # Get corresponding annotation file
                ann_file = ann_path / f"{img_file.name}.json"
                if not ann_file.exists():
                    print(f"Annotation file not found: {ann_file}")
                    continue

                # Load image and annotations
                image = cv2.imread(str(img_file))
                if image is None:
                    print(f"Image not found or could not be loaded: {img_file}")
                    continue
                components = self.load_annotation(ann_file)

                # Process each component in the image
                for component in components:
                    if component['class'] not in self.class_mapping:
                        continue

                    # Extract and preprocess component
                    comp_image = self.extract_component(image, component['bbox'])

                    # Normalize
                    comp_image = comp_image.astype(np.float32) / 255.0

                    # Create one-hot encoded label
                    label = np.zeros(self.num_classes)
                    label[self.class_mapping[component['class']]] = 1

                    batch_images.append(comp_image)
                    batch_labels.append(label)

                    if len(batch_images) == batch_size:
                        # print(f"Yielding batch of size: {len(batch_images)}")
                        yield np.array(batch_images), np.array(batch_labels)
                        batch_images = []
                        batch_labels = []
            # Yield remaining samples
            if batch_images:
                print(f"Yielding remaining batch of size: {len(batch_images)}")
                yield np.array(batch_images), np.array(batch_labels)
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),

                tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32)
            )
        )
        # Repeat the dataset for multiple epochs
        return dataset.repeat().prefetch(tf.data.AUTOTUNE) # If we don't want repeating, we can simply remove the .repeat()