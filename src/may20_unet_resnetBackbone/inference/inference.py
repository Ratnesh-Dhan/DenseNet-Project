import tensorflow as tf
import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input

def load_image(image_path):
    IMG_SIZE = (256, 256)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image

def predict_mask(image_path, save_output=True):
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)

    prediction = model.predict(image)

    plt.imshow(prediction[0])
    plt.axis('off')
    plt.title("Prediction")
    plt.show()

if __name__ == "__main__":
    model_path = "../models/unet_resnet50_corrosion.h5"

    if os.path.exists(model_path):
        print("\nModel's file location exists.\n")
    else:
        print("\nModel's file location does not exist. Exiting...\n")
        sys.exit(0)

    model = tf.keras.models.load_model(model_path, compile=False)

    # image_path = r"D:\NML ML Works\kaggle_semantic_segmentation_CORROSION_dataset\validate\images\img72_aug1.jpg"
    image_path = r"D:\NML 2nd working directory\Modified\CSS"
    image_path = os.path.join(image_path, "CSS_AG_3386.jpg")
    predict_mask(image_path)