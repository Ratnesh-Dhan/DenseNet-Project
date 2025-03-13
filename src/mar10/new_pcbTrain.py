#  https://github.com/facebookresearch/segment-anything
# this is SAM for image segmentation . not the code but the above link. 
# The below code is to train object-detection from pcbDataset

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2
# from sklearn.model_selection import train_test_split

# Define paths
TRAIN_DIR = '../../Datasets/pcbDataset/train'
VAL_DIR = '../../Datasets/pcbDataset/validation'
IMG_SIZE = (416, 416)  # Common size for object detection
NUM_CLASSES = 9  # From your meta.json

# Load class mapping from meta.json
def load_class_mapping(meta_path='../../Datasets/pcbDataset/meta.json'):
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # Create id to index mapping
    id_to_index = {}
    class_names = []
    
    for idx, class_info in enumerate(meta_data['classes']):
        id_to_index[class_info['id']] = idx
        class_names.append(class_info['title'])
    
    return id_to_index, class_names

# Parse annotations from JSON files
def parse_annotation(annotation_path, id_to_index):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    img_height = data['size']['height']
    img_width = data['size']['width']
    
    boxes = []
    classes = []
    
    for obj in data['objects']:
        # Get class index
        class_id = obj['classId']
        class_index = id_to_index[class_id]
        
        # Get normalized box coordinates (xmin, ymin, xmax, ymax)
        points = obj['points']['exterior']
        xmin = points[0][0] / img_width
        ymin = points[0][1] / img_height
        xmax = points[1][0] / img_width
        ymax = points[1][1] / img_height
        
        # Store normalized coordinates
        boxes.append([xmin, ymin, xmax, ymax])
        classes.append(class_index)
    
    return np.array(boxes), np.array(classes)

# Load dataset
def load_dataset(base_dir, id_to_index):
    image_paths = []
    all_boxes = []
    all_classes = []
    
    img_dir = os.path.join(base_dir, 'img')
    ann_dir = os.path.join(base_dir, 'ann')
    
    for filename in os.listdir(img_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Image path
            img_path = os.path.join(img_dir, filename)
            
            # Annotation path (assuming same name with .json extension)
            ann_filename = f'{filename}.json'
            ann_path = os.path.join(ann_dir, ann_filename)
            
            if os.path.exists(ann_path):
                boxes, classes = parse_annotation(ann_path, id_to_index)
                
                if len(boxes) > 0:  # Only add if there are annotations
                    image_paths.append(img_path)
                    all_boxes.append(boxes)
                    all_classes.append(classes)
    
    return image_paths, all_boxes, all_classes

# Data generator for training
def data_generator(image_paths, all_boxes, all_classes, batch_size=8, max_objects=10):
    num_samples = len(image_paths)
    indices = np.arange(num_samples)
    
    while True:
        np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            actual_batch_size = len(batch_indices)  # In case we have a partial batch at the end
            
            # Pre-allocate arrays with correct shapes
            batch_images = np.zeros((actual_batch_size, *IMG_SIZE, 3), dtype=np.float32)
            batch_targets = np.zeros((actual_batch_size, max_objects, 5 + NUM_CLASSES), dtype=np.float32)
            
            for i, idx in enumerate(batch_indices):
                # Load and preprocess image
                img = cv2.imread(image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype(np.float32) / 255.0
                
                # Store the image
                batch_images[i] = img
                
                # Get boxes and classes for this image
                boxes = all_boxes[idx]
                classes = all_classes[idx]
                
                # Fill in actual objects (up to max_objects)
                num_boxes = min(len(boxes), max_objects)
                for j in range(num_boxes):
                    box = boxes[j]
                    cls = classes[j]
                    
                    # Convert from [xmin, ymin, xmax, ymax] to [x_center, y_center, width, height]
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    
                    batch_targets[i, j, 0] = x_center
                    batch_targets[i, j, 1] = y_center
                    batch_targets[i, j, 2] = width
                    batch_targets[i, j, 3] = height
                    batch_targets[i, j, 4] = 1.0  # Object confidence
                    batch_targets[i, j, 5 + cls] = 1.0  # Class one-hot
            
            yield batch_images, batch_targets

# Custom loss function for object detection
# def detection_loss(y_true, y_pred):
#     # Make sure both tensors have the same shape
#     shape = tf.shape(y_pred)
#     y_true = tf.reshape(y_true, shape)
    
#     # Object confidence loss
#     obj_mask = y_true[..., 4:5]
#     obj_loss = tf.keras.losses.binary_crossentropy(obj_mask, y_pred[..., 4:5])
#     obj_loss = tf.reduce_sum(obj_loss) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
#     # Class prediction loss
#     class_loss = tf.keras.losses.categorical_crossentropy(
#         y_true[..., 5:], y_pred[..., 5:], from_logits=False
#     )
#     class_loss = tf.reduce_sum(class_loss * obj_mask) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
#     # Bounding box coordinates loss
#     xy_loss = tf.reduce_sum(
#         tf.square(y_true[..., 0:2] - y_pred[..., 0:2]) * obj_mask
#     ) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
#     wh_loss = tf.reduce_sum(
#         tf.square(tf.sqrt(y_true[..., 2:4]) - tf.sqrt(y_pred[..., 2:4])) * obj_mask
#     ) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
#     # Total loss
#     total_loss = obj_loss + class_loss + xy_loss + wh_loss
    
#     return total_loss

# Custom loss function for object detection
def detection_loss(y_true, y_pred):
    # Make sure both tensors have the same shape
    shape = tf.shape(y_pred)
    y_true = tf.reshape(y_true, shape)
    
    # Object confidence loss (confidence score)
    obj_mask = y_true[..., 4:5]  # Shape: [batch, max_objects, 1]
    obj_loss = tf.keras.losses.binary_crossentropy(obj_mask, y_pred[..., 4:5])
    obj_loss = tf.reduce_sum(obj_loss) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
    # Class prediction loss
    class_loss = tf.keras.losses.categorical_crossentropy(
        y_true[..., 5:], y_pred[..., 5:], from_logits=False
    )
    # Expand dimensions to match
    class_loss = tf.reduce_sum(class_loss * tf.squeeze(obj_mask, axis=-1)) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
    # Bounding box coordinate losses - handle each coordinate separately
    # X and Y coordinates (center)
    xy_diff = tf.square(y_true[..., 0:2] - y_pred[..., 0:2])
    # Reduce along the coordinate dimension to match obj_mask shape
    xy_loss = tf.reduce_sum(
        tf.reduce_sum(xy_diff, axis=-1, keepdims=True) * obj_mask
    ) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
    # Width and height
    wh_diff = tf.square(
        tf.sqrt(tf.maximum(y_true[..., 2:4], 1e-10)) - 
        tf.sqrt(tf.maximum(y_pred[..., 2:4], 1e-10))
    )
    # Reduce along the coordinate dimension to match obj_mask shape
    wh_loss = tf.reduce_sum(
        tf.reduce_sum(wh_diff, axis=-1, keepdims=True) * obj_mask
    ) / tf.maximum(tf.reduce_sum(obj_mask), 1.0)
    
    # Total loss
    total_loss = obj_loss + class_loss + xy_loss + wh_loss
    
    return total_loss
    
# Build a simple object detection model
def build_model(input_shape, max_objects, num_classes):
    # Use MobileNetV2 as base model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Add detection heads
    x = base_model.output
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Flatten the output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layer to get to the right number of elements
    dense_output_size = max_objects * (5 + num_classes)
    x = tf.keras.layers.Dense(dense_output_size)(x)
    
    # Reshape to the final output dimensions
    output = tf.keras.layers.Reshape((max_objects, 5 + num_classes))(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        loss=detection_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    
    return model

# Function to visualize predictions
def visualize_predictions(model, image_paths, id_to_index, class_names, num_samples=5):
    reverse_class_map = {v: k for k, v in id_to_index.items()}
    
    # Randomly select some images
    indices = np.random.choice(len(image_paths), num_samples, replace=False)
    
    plt.figure(figsize=(15, 15))
    
    for i, idx in enumerate(indices):
        # Load and preprocess image
        img_path = image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # Resize for model input
        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Make prediction
        prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0]
        
        # Filter out valid predictions (with confidence > threshold)
        confidence_threshold = 0.5
        valid_indices = np.where(prediction[:, 4] > confidence_threshold)[0]
        
        # Create a copy for drawing
        img_with_boxes = img.copy()
        
        for j in valid_indices:
            pred = prediction[j]
            
            # Get bounding box coordinates
            x_center, y_center, width, height = pred[:4]
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int((x_center - width/2) * img_width)
            y_min = int((y_center - height/2) * img_height)
            x_max = int((x_center + width/2) * img_width)
            y_max = int((y_center + height/2) * img_height)
            
            # Get class with highest probability
            class_idx = np.argmax(pred[5:])
            class_name = class_names[class_idx]
            confidence = pred[4]
            
            # Draw rectangle and label
            cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img_with_boxes, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display image with predictions
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img_with_boxes)
        plt.title(f"Predictions for {os.path.basename(img_path)}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main function to run the entire pipeline
def main():
     # Load class mapping
    id_to_index, class_names = load_class_mapping()
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load datasets
    train_images, train_boxes, train_classes = load_dataset(TRAIN_DIR, id_to_index)
    val_images, val_boxes, val_classes = load_dataset(VAL_DIR, id_to_index)
    
    print(f"Loaded {len(train_images)} training images and {len(val_images)} validation images")
    
    # Find max objects in any image for model design
    max_objects_train = max(len(boxes) for boxes in train_boxes)
    max_objects_val = max(len(boxes) for boxes in val_boxes)
    max_objects = max(max_objects_train, max_objects_val)
    print(f"Maximum objects in any image: {max_objects}")
    
    # Create data generators
    batch_size = 8
    train_gen = data_generator(train_images, train_boxes, train_classes, batch_size, max_objects)
    val_gen = data_generator(val_images, val_boxes, val_classes, batch_size, max_objects)
    
    # Build model with the same max_objects value
    input_shape = (*IMG_SIZE, 3)
    model = build_model(input_shape, max_objects, NUM_CLASSES)
    model.summary()
    
    # Configure callbacks
    callbacks = [
        # EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint('pcb_detector.h5', save_best_only=True, verbose=1)
    ]
    
    # Train model
    steps_per_epoch = len(train_images) // batch_size
    validation_steps = max(1, len(val_images) // batch_size)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=100,  # Adjust as needed
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 1, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Visualize some predictions
    visualize_predictions(model, val_images, id_to_index, class_names)
    
    # Save the model
    model.save('pcb_detector_final.h5')
    print("Model saved as 'pcb_detector_final.h5'")

# def main():
#     # Load class mapping
#     id_to_index, class_names = load_class_mapping()
#     print(f"Loaded {len(class_names)} classes: {class_names}")
    
#     # Load datasets
#     train_images, train_boxes, train_classes = load_dataset(TRAIN_DIR, id_to_index)
#     val_images, val_boxes, val_classes = load_dataset(VAL_DIR, id_to_index)
    
#     print(f"Loaded {len(train_images)} training images and {len(val_images)} validation images")
    
#     # Find max objects in any image for model design
#     max_objects_train = max(len(boxes) for boxes in train_boxes)
#     max_objects_val = max(len(boxes) for boxes in val_boxes)
#     max_objects = max(max_objects_train, max_objects_val)
#     print(f"Maximum objects in any image: {max_objects}")
    
#     # Create data generators
#     batch_size = 8
#     train_gen = data_generator(train_images, train_boxes, train_classes, batch_size, max_objects)
#     val_gen = data_generator(val_images, val_boxes, val_classes, batch_size, max_objects)
    
#     # Build model with the same max_objects value
#     input_shape = (*IMG_SIZE, 3)
#     model = build_model(input_shape, max_objects, NUM_CLASSES)
#     model.summary()
    
#     # Rest of the function remains the same...

if __name__ == "__main__":
    main()