import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model("mask_rcnn_model.keras")

def load_and_process_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((512,512))
    image = np.array(image)/255.0 # Normalizing
    image = np.expand_dims(image, axis=0) # Adding batch dimension
    return image

def visualize(og_img, img):
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(og_img))
    plt.title("og image")
    plt.axis("off")

    plt.subplot(1,2,2)
    # plt.imshow(img, cmap='jet', alpha=0.7)
    plt.imshow(img)
    plt.title("result from ml")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "../../Datasets/testDataset/img/2007_007948.jpg"
    image = load_and_process_image(image_path)
    print(image.shape)
    predicted_mask = model.predict(image)

    visualize(image, predicted_mask)
    # Post-process the predicted mask
    predicted_mask = np.argmax(predicted_mask[0], axis=-1)  # Convert to (512, 512) with class indices
