import tensorflow as tf
import cv2

model = tf.keras.models.load_model("object_detection_model.keras")

image =  cv2.imread(r"C:\Users\NDT Lab\Pictures\dog.jpg")
print(image.shape)
image = cv2.resize(image, (224, 224))/255.0
image = image.reshape(1, 224, 224, 3)
print(image.shape)

predicted_box , predicted_class = model.predict(image)
print(predicted_class)
print(predicted_box)
