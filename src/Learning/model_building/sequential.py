import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a Sequential model
model = keras.Sequential([
    # layers.Input(shape=(784,)),  # Input layer, e.g., for flattened 28x28 images
    layers.Input(shape=(28, 28)),  # Input layer, e.g., for flattened 28x28 images
    layers.Conv2D(32, 3, activation='relu'),  # First convolutional layer with 32 filters and ReLU activation
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 units and ReLU activation
    layers.Dropout(0.2),  # Dropout layer for regularization
    layers.Dense(10, activation='softmax')  # Output layer with 10 units for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()