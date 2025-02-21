import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from typing import List, Tuple, Dict
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import json
import os

class PCBDataset:
    """
    Custom dataset handler for PCB component data
    """

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.class_mapping = {
            'Cap1': 0, 'Cap2': 1, 'Cap3': 2, 'Cap4': 3, 'MOSFET': 4, 'Mov': 5, 'Resistor': 6, 'Transformer': 7
        }
        self.num_classes = len(self.class_mapping)

    def load_annotation(self, json_path: Path) -> List[Dict]:
        """ Load and parse annotation JSON file
            Return list of components with their bounding boxes"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        components = []
        for obj in data['objects']:
            # Extract component information
            component = {
                'class': obj['classTitle'],
                'bbox': obj['points']['exterior'],
                'image_size': (data['size']['width'], data['size']['height'])
            }
            components.append(component)
        return components
    
    def extract_component(self, image: np.ndarray, bbox: List[List[int]]) -> np.ndarray:
        """
        Extract and preprocss a component using its bounding box
        """
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
        # Resize to standard size
        image = cv2.resize(image, (224, 224))
        return image  # WE ARE RETURNING HERE TO REDUCE THE COMPOUTATOINAL LOAD AND UNNESSESSARY PREPROCESS
        
    
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
    
    def count_samples(self, split: str) -> int:
        """Count the number of samples in the specified dataset split."""
        split_path = self.base_path / split
        img_path = split_path / 'img'
        ann_path = split_path / 'ann'
        total_samples = 0
        images = list(img_path.glob('*.jpg'))    
        for img_file in images:
            ann_file = ann_path / f"{img_file.name}.json"
            if not ann_file.exists():
                continue
            components = self.load_annotation(ann_file)
            total_samples += sum(1 for comp in components if comp['class'] in self.class_mapping)
        return total_samples

# export TF_ENABLE_ONEDNN_OPTS=0  # For Linux
# set TF_ENABLE_ONEDNN_OPTS=0     # For Windows

# Set threading options
tf.config.threading.set_intra_op_parallelism_threads(12)  # Number of threads for parallelism within an individual operation
tf.config.threading.set_inter_op_parallelism_threads(6)   # Number of threads for parallelism between independent operations

# Set environment variables to optimize performance
os.environ['OMP_NUM_THREADS'] = '12'  # OpenMP threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '12'
os.environ['TF_NUM_INTEROP_THREADS'] = '6'


#4. Enable XLA (Accelerated Linear Algebra)
#XLA can improve performance by compiling subgraphs into optimized kernels:
tf.config.optimizer.set_jit(True)  # Enable XLA

def create_and_train_model(dataset_path: str, batch_size: int = 32, epochs: int = 30):
    """Create and Train model"""

    # Initialize dataset
    pcb_dataset = PCBDataset(dataset_path)

    # Calculate the total steps per epoch 
    train_samples = pcb_dataset.count_samples('train')
    val_samples = pcb_dataset.count_samples('validation')
    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size

    # Create Datasets
    train_dataset = pcb_dataset.create_data_generator('train', batch_size)
    val_dataset = pcb_dataset.create_data_generator('validation', batch_size)

    # Create Model
    model = create_model(pcb_dataset.num_classes)

    # Train model
    history = model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs=epochs,
        steps_per_epoch= steps_per_epoch,  # We are adding this to see the total number of steps per epoch in the termianl
        validation_steps = validation_steps, # Same reason as the above one
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5),
            tf.keras.callbacks.ModelCheckpoint('best_pcb_model.h5', save_best_only=True)
        ]
    )
    return model , history

def create_model(num_classes: int):
    """Create the model architecture"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Fine-tune the last 30 layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

if __name__ == "__main__":
    dataset_path = "../../Datasets/pcbDataset"
    model, history = create_and_train_model(dataset_path=dataset_path,batch_size=32 , epochs=20)

    # Save the model
    model.save('pcb_component_classifier.h5')
    model.save('pcb_component_classifier.keras')