import tensorflow as tf
import numpy as np, json, os, io, base64, matplotlib.pyplot as plt, sys
from tensorflow.keras import layers, Model
from PIL import Image

# Constants
IMG_SIZE = 512
BATCH_SIZE = 2  # Smaller batch size due to memory requirements
NUM_CLASSES = 21  # Including background
EPOCHS = 50

def decode_bitmap_mask(bitmap_data, origin, height, width):
    """Decode base64 bitmap data to numpy array mask."""
    decoded_data = base64.b64decode(bitmap_data)
    bitmap_img = Image.open(io.BytesIO(decoded_data))
    bitmap_array = np.array(bitmap_img)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y = origin
    h, w = bitmap_array.shape[:2]
    mask[y:y+h, x:x+w] = bitmap_array[:, :, 3] > 0  # Use alpha channel
    
    return mask

def process_example(image_path, annotation_path):
    """Process a single image and its annotations."""
    print(annotation_path)
    sys.exit(0)
    # Convert annotation_path to string
    annotation_path = tf.strings.to_string(annotation_path)  # {{ edit_1 }}
    
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    
    # Read annotation
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    # Get original dimensions
    orig_height = annotation['size']['height']
    orig_width = annotation['size']['width']
    
    # Process each object instance
    masks = []
    boxes = []
    class_ids = []
    
    for obj in annotation['objects']:
        if obj['geometryType'] == 'bitmap':
            # Create instance mask
            mask = decode_bitmap_mask(
                obj['bitmap']['data'],
                obj['bitmap']['origin'],
                orig_height,
                orig_width
            )
            
            # Get bounding box
            x, y = obj['bitmap']['origin']
            non_zero = np.nonzero(mask)
            if len(non_zero[0]) > 0:  # Check if mask is not empty
                y1, x1 = np.min(non_zero, axis=1)
                y2, x2 = np.max(non_zero, axis=1)
                
                # Normalize coordinates
                x1 = x1 / orig_width
                y1 = y1 / orig_height
                x2 = x2 / orig_width
                y2 = y2 / orig_height
                
                # Resize mask to standard size
                mask = tf.image.resize(
                    mask[..., np.newaxis],
                    [IMG_SIZE, IMG_SIZE],
                    method='nearest'
                )
                
                masks.append(mask)
                boxes.append([x1, y1, x2, y2])
                class_ids.append(obj['classId'])
    
    # Convert to tensors
    if masks:
        masks = tf.stack(masks)
        boxes = tf.constant(boxes, dtype=tf.float32)
        class_ids = tf.constant(class_ids, dtype=tf.int32)
    else:
        # Handle images with no objects
        masks = tf.zeros([1, IMG_SIZE, IMG_SIZE, 1], dtype=tf.float32)
        boxes = tf.zeros([1, 4], dtype=tf.float32)
        class_ids = tf.zeros([1], dtype=tf.int32)
    
    return image, (masks, boxes, class_ids)

def create_mask_rcnn_model():
    """Create Mask R-CNN model architecture."""
    # Use ResNet50 as backbone
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Feature Pyramid Network
    C2 = backbone.get_layer('conv2_block3_out').output
    C3 = backbone.get_layer('conv3_block4_out').output
    C4 = backbone.get_layer('conv4_block6_out').output
    C5 = backbone.get_layer('conv5_block3_out').output
    
    # FPN layers
    P5 = layers.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = layers.Add(name="fpn_p4add")([
        layers.UpSampling2D(size=(2, 2))(P5),
        layers.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)
    ])
    P3 = layers.Add(name="fpn_p3add")([
        layers.UpSampling2D(size=(2, 2))(P4),
        layers.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = layers.Add(name="fpn_p2add")([
        layers.UpSampling2D(size=(2, 2))(P3),
        layers.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)
    ])
    
    # RPN (Region Proposal Network)
    rpn = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(P2)
    rpn_class = layers.Conv2D(2, (1, 1), activation='sigmoid')(rpn)
    rpn_bbox = layers.Conv2D(4, (1, 1))(rpn)
    
    # ROI Pooling
    roi_pool = layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    )(P2)
    
    # Box head
    bbox_features = layers.Dense(1024, activation='relu')(roi_pool)
    bbox_features = layers.Dense(1024, activation='relu')(bbox_features)
    bbox_pred = layers.Dense(NUM_CLASSES * 4)(bbox_features)
    
    # Class head
    class_features = layers.Dense(1024, activation='relu')(roi_pool)
    class_pred = layers.Dense(NUM_CLASSES, activation='softmax')(class_features)
    
    # Mask head
    mask_features = layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu')(P2)
    mask_features = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(mask_features)
    mask_pred = layers.Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid')(mask_features)
    
    model = Model(
        inputs=backbone.input,
        outputs=[rpn_class, rpn_bbox, bbox_pred, class_pred, mask_pred]
    )
    
    return model

def mask_rcnn_loss():
    """Custom loss function for Mask R-CNN."""
    def loss(y_true, y_pred):
        # Extract components
        true_masks, true_boxes, true_classes = y_true
        pred_rpn_class, pred_rpn_bbox, pred_bbox, pred_class, pred_masks = y_pred
        
        # RPN losses
        rpn_class_loss = tf.keras.losses.binary_crossentropy(
            true_classes,
            pred_rpn_class
        )
        rpn_bbox_loss = tf.keras.losses.huber(
            true_boxes,
            pred_rpn_bbox
        )
        
        # Detection losses
        class_loss = tf.keras.losses.sparse_categorical_crossentropy(
            true_classes,
            pred_class
        )
        bbox_loss = tf.keras.losses.huber(
            true_boxes,
            pred_bbox
        )
        
        # Mask loss
        mask_loss = tf.keras.losses.binary_crossentropy(
            true_masks,
            pred_masks
        )
        
        return rpn_class_loss + rpn_bbox_loss + class_loss + bbox_loss + mask_loss
    
    return loss

def create_dataset(image_paths, annotations_dir):
    """Create training dataset."""
    def get_annotation_path(image_path):
        image_name = tf.strings.split(image_path, os.sep)[-1]
        return tf.strings.join([annotations_dir, image_name, '.json'], '')
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    annotation_paths = dataset.map(get_annotation_path)
    dataset = tf.data.Dataset.zip((dataset, annotation_paths))
    dataset = dataset.map(process_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_model():
    """Train the Mask R-CNN model."""
    # Setup paths
    dataset_dir = '../../Datasets/testDataset/'
    image_dir = os.path.join(dataset_dir, 'img')
    annotations_dir = os.path.join(dataset_dir, 'ann')
    
    # Get image paths
    image_paths = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))
    
    # Split dataset
    np.random.shuffle(image_paths)
    split_idx = int(len(image_paths) * 0.8)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Create datasets
    train_dataset = create_dataset(train_paths, annotations_dir)
    val_dataset = create_dataset(val_paths, annotations_dir)
    
    # Create and compile model
    model = create_mask_rcnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=mask_rcnn_loss(),
        metrics=['accuracy']
    )
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_mask_rcnn_model.h5',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    # Train model
    model, history = train_model()
    
    # Save model
    model.save('mask_rcnn_model.keras')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')