import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from trasferLearningModelHardMode import build_unet_with_transfer_learning
model = build_unet_with_transfer_learning()
print(model.output_shape)
import sys
sys.exit(0)

# Load trained model
model = tf.keras.models.load_model('./model/unet_resnet50_multiclass.h5')

# Load and preprocess image
def preprocess_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Post-process prediction to get class mask
def postprocess_prediction(prediction):
    predicted_mask = np.argmax(prediction[0], axis=-1)  # shape: (H, W)
    return predicted_mask

# Example usage
img_base_location = "../test"
image_path = os.path.join(img_base_location, "1.png")
input_image = preprocess_image(image_path)
prediction = model.predict(input_image)
print("prediction: ", prediction)
mask = postprocess_prediction(prediction)

# Optional: visualize input and mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(Image.open(image_path))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Predicted Mask')
# plt.imshow(mask, cmap='jet')  # or use custom colormap for 3 classes
plt.imshow(mask)  # or use custom colormap for 3 classes
plt.axis('off')
plt.show()
