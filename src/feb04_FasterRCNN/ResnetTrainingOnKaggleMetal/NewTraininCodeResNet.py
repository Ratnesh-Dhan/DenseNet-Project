import tensorflow as tf
import os
import numpy as np
# from sklearn.model_selection import train_test_split

# Set TensorFlow to use CPU optimizations
tf.config.threading.set_intra_op_parallelism_threads(12)  # Your CPU threads
tf.config.threading.set_inter_op_parallelism_threads(6)   # Your CPU cores

# Define model architecture with ResNet50
def create_model():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)  # ResNet's standard input size
    )
    
    # Freeze early layers for transfer learning
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')  # 6 classes for defects
    ])
    
    return model

# Data preprocessing function
def preprocess_data(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

# Create data pipeline optimized for CPU
def create_dataset(images, labels, batch_size=32, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = (dataset
        .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
    
    return dataset

# Training function with model saving
def train_model(train_dataset, val_dataset, epochs=50):
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for model saving and early stopping
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        # worker=6,  # Number of CPU cores
        # use_multiprocessing=True
    )
    
    # Save final model
    model.save('final_model.h5')
    
    return model, history

# Function to load dataset paths and labels
def load_dataset(base_path, split_folder):
    image_paths = []
    labels = []
    class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(base_path, split_folder, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} does not exist. Skipping.")
            continue
        
        for img_name in os.listdir(class_path):
            if img_name.endswith('.bmp') or img_name.endswith('.png'):
                image_path = os.path.join(class_path, img_name)
                if os.path.isfile(image_path):  # Ensure it's a valid file
                    image_paths.append(image_path)
                    labels.append(idx)
                else:
                    print(f"Warning: {image_path} is not a valid file. Skipping.")
    
    # Convert to numpy arrays for compatibility with TensorFlow
    image_paths = np.array(image_paths, dtype=np.str_)
    labels = np.array(labels, dtype=np.int32)
    
    return image_paths, labels

# Main execution
def main():
    # Base path to the dataset
    base_path = 'NEU Metal Surface Defects Data'
    
    # Load training and validation data
    train_images, train_labels = load_dataset(base_path, 'train')
    val_images, val_labels = load_dataset(base_path, 'valid')
    
    # Debug: Print first 5 image paths and labels
    print("Sample training image paths:", train_images[:5])
    print("Sample training labels:", train_labels[:5])
    
    # Create datasets
    train_dataset = create_dataset(train_images, train_labels)
    val_dataset = create_dataset(val_images, val_labels, is_training=False)
    
    # Train model
    model, history = train_model(train_dataset, val_dataset)
    
    return model, history

if __name__ == "__main__":
    model, history = main()