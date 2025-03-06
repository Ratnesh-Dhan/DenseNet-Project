# https://www.tensorflow.org/tutorials
# https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf

# loading dataset
mnist = tf.keras.datasets.mnist
print("Tensorflow version:", tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions
# Defining loss function
lossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lossFunction(y_train[:1], predictions).numpy()