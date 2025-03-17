import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the saved model
model = tf.keras.models.load_model('my_model.keras')

ary = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Step 2: Load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Decode the image
    image = tf.image.resize(image, [32, 32])  # Resize to match model input
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Replace 'path_to_your_image.jpg' with the actual image path
image_path = '../img/dog2.jpg'
new_image = load_and_preprocess_image(image_path)

# Step 3: Make predictions
predictions = model.predict(new_image)
pred_label = tf.argmax(predictions, axis=1)

# Print the predicted label
print("Predicted label:", pred_label.numpy())
print("Thing detected: ", ary[pred_label.numpy()[0]])