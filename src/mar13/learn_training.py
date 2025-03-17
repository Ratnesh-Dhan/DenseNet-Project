"""
https://www.geeksforgeeks.org/how-to-train-tensorflow-models-in-python/
https://www.learnpytorch.io/00_pytorch_fundamentals/
"""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset_name = 'cifar10'
(train_dataset, test_dataset), dataset_info = tfds.load(
    name=dataset_name,
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True
)

image, label = next(iter(train_dataset.take(1)))
print("image shape: ",image.shape)
print("label: ", label)

num_classes = dataset_info.features['label'].num_classes
print(num_classes)

# Preprocessing & Normalization
def preprocessing_data(image, label):
    image = tf.cast(image,tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocessing_data)
test_dataset = test_dataset.map(preprocessing_data)

input_dim = (32, 32, 3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_dim),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Compiling model 
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

batch_size = 128
num_epochs = 30

# To process the dataset in batches create the batches of batch_size 
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)

# Train the model
model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
model.save("my_model.keras")

# Evaluate
loss, accuracy = model.evaluate(test_dataset)
print("Test Loss: ", loss)
print("Test Accuracy: ", accuracy)

# Time to predict with this model
# create a custom array of image size
new_image = tf.constant(np.random.rand(32, 32, 3), dtype=tf.float64)
# Extend the dimension 4D
new_image = tf.expand_dims(new_image, axis=0)
 
# Prediction
predictions = model.predict(new_image)
# predicted label
pred_label = tf.argmax(predictions, axis =1)
print(pred_label.numpy())
# tf.keras.saving.save_model(model, 'my_model.keras')