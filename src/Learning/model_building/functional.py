import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define input layer
inputs = keras.Input(shape=(32, 32, 3)) # e.g., for 32x32 color images

# Define layers and connect them
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()


# -> Compile the Model: Specify the optimizer, loss function, and metrics to be used during training.
# -> Train the Model: Use model.fit() with your training data and labels.
# -> Evaluate the Model: Use model.evaluate() to assess performance on a test set.
# -> Make Predictions: Use model.predict() to get predictions on new data.
# -> Save the Model: Use model.save() to store the trained model for later use.
