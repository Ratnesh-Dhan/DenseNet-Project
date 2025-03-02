import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from PCBDataset import PCBDataset
import os


# export TF_ENABLE_ONEDNN_OPTS=0  # For Linux
# set TF_ENABLE_ONEDNN_OPTS=0     # For Windows

# Set threading options
# tf.config.threading.set_intra_op_parallelism_threads(12)  # Number of threads for parallelism within an individual operation
# tf.config.threading.set_inter_op_parallelism_threads(6)   # Number of threads for parallelism between independent operations

# Set environment variables to optimize performance
os.environ['OMP_NUM_THREADS'] = '12'  # OpenMP threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '12'
os.environ['TF_NUM_INTEROP_THREADS'] = '6'

# Set the global mixed precision policy to 'mixed_float16'
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
mixed_precision.set_global_policy('float32')

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
    model, history = create_and_train_model(dataset_path=dataset_path,batch_size=32 , epochs=15)

    # Save the model
    model.save('pcb_component_classifier.h5')
    model.save('pcb_component_classifier.keras')