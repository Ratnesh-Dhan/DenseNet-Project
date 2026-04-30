import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from model import dice_loss

model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={"dice_loss": dice_loss}
)

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    print("\n**********************************************\n")
    print("min:", pred.min(), "max:", pred.max(), "mean:", pred.mean())
    print("\n**********************************************\n")

    mask = (pred > 0.1).astype(np.uint8)

    return mask

def show_prediction_mask():
    img_path = "test.png"

    mask = predict_image(img_path)

    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title("Predicted Mask")
    plt.show()

def overlay(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256,256))

    mask = predict_image(image_path).squeeze()

    img_resized[mask == 1] = [0, 255, 0]  # green overlay
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imwrite(path, image)

if __name__ == "__main__":
    image_path = "/mnt/z/DATASETS/corrosion/000007.jpg"
    overlay(image_path)