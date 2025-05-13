import tensorflow as tf
import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def save_binary_mask(mask, count, target_class=2, output_dir=r"D:\NML ML Works\Testing_mask_binary"):
    os.makedirs(output_dir, exist_ok=True)
    # Create binary mask
    binary_mask = np.where(mask == target_class, 255, 0).astype(np.uint8)
    # Save as PNG
    save_path = os.path.join(output_dir, f'binary_mask_{count}.png')
    cv2.imwrite(save_path, binary_mask)


def load_and_predict(model, image_path, count):
    # Load model
    
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
    # save_results(og_image, segmentation_mask, prediction[0], count=count)
    save_binary_mask(segmentation_mask, count)

    return segmentation_mask, prediction

def save_results(original, mask, pred_prob, count):
    # Convert original to uint8
    original_uint8 = original.astype('uint8')

    # Create a blank RGBA image for overlay
    overlay_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

    # Apply neon color with transparency only to class 2 (corrosion)
    neon = [57, 255, 50, 150]  # R, G, B, Alpha (150 = semi-transparent)
    overlay_rgba[mask == 2] = neon  # neon for corrosion

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
    save_path = os.path.join(r"D:\NML ML Works\Testing_mask", f'segmentation_result_{count}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    model_path=r'C:\Users\NDT Lab\Software\DenseNet-Project\DenseNet-Project\src\april24_corrosion\train\model\unet_resnet50_multiclass.h5'
    model = load_model(model_path)
    # locale = r"D:\NML ML Works\corrosionDataset\images"
    locale = r"D:\NML ML Works\Testing"
    files = os.listdir(locale)
    # half = int(len(files)/2)
    # files = files[half :]
    count = 1
    for f in files:
        f = os.path.join(str(locale), f)
        print(f)
        load_and_predict(
            model,
            image_path=f,
            count=count
        )
        count = count +1
        print("Done")
# This function needs to 