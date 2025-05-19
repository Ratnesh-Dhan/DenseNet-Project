import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_predict(model_path, image_path):
    # Load model
    model = load_model(model_path, compile=False)
    print(f"Model loaded: input shape {model.input_shape}, output shape {model.output_shape}")
    
    # Load and preprocess image - USING IMAGENET NORMALIZATION
    img = load_img(image_path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # Saving the original image for matplot show
    og_image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    og_image = cv2.resize(og_image, (512, 512))
    
    # Run prediction
    prediction = model.predict(img_batch)
    
    # Get segmentation mask
    segmentation_mask = np.argmax(prediction[0], axis=-1)
    
    # Show results
    display_results(og_image, segmentation_mask, prediction[0])
    
    return segmentation_mask, prediction


# def display_results(original, mask, pred_prob):
#     # Define colors for visualization
#     colors = [
#         # [0, 0, 0],    # Red for class 0
#         # [0, 0, 0],    # Green for class 1
#         # [0, 0, 255]     # Blue for class 2
#         [255, 255, 255],    # Red for class 0
#         [255, 255, 255],    # Green for class 1
#         [0, 0, 255]     # Blue for class 2
#     ]

#     # Create colored mask
#     colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#     for i, color in enumerate(colors):
#         colored_mask[mask == i] = color

#     # Convert original to uint8
#     original_uint8 = original.astype('uint8')

#     # Blend the original and mask using alpha
#     alpha = 0.5  # Transparency factor
#     overlay = cv2.addWeighted(original_uint8, 1 - alpha, colored_mask, alpha, 0)

#     # Display
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original_uint8)
#     plt.title('Original Image')
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.imshow(overlay)
#     plt.title('Overlay Segmentation')
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

def display_results(original, mask, pred_prob):
    # Convert original to uint8
    original_uint8 = original.astype('uint8')

    # Create a blank RGBA image for overlay
    overlay_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

    # Apply blue color with transparency only to class 2 (corrosion)
    # blue = [0, 0, 255, 150]  # R, G, B, Alpha (150 = semi-transparent)
    neon = [57, 255, 50, 150]  # R, G, B, Alpha (150 = semi-transparent)
    overlay_rgba[mask == 2] = neon  # neon for corrosion
    # overlay_rgba[mask == 2] = blue  # class 2 is corrosion

    # Convert original to RGBA
    original_rgba = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2RGBA)

    # Overlay mask on original using alpha blending
    overlay_result = original_rgba.copy()
    alpha_mask = overlay_rgba[:, :, 3:] / 255.0
    overlay_result[:, :, :3] = (
        overlay_result[:, :, :3] * (1 - alpha_mask) + overlay_rgba[:, :, :3] * alpha_mask
    ).astype(np.uint8)

    # Display
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_uint8)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_result)
    plt.title('Corrosion Highlighted (Class 2)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_and_predict(
        model_path='./model/may_16_unet_resnet50_multiclass.h5',
        image_path='../test/image/1.png'
        # image_path=r'D:\NML ML Works\Testing\WhatsApp Image 2025-05-09 at 3.29.04 PM.jpeg'
    )
# This function needs to 

