import tensorflow as tf
from PIL import Image
model = tf.keras.models.load_model("model.keras")
img = Image.open("../img/1.png")

