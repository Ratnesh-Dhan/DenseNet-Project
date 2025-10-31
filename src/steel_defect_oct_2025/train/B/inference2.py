import tensorflow as tf
import numpy as np
import cv2
import os

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def load_and_preprocess_image(image_path, target_size=(200, 200)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return tf.expand_dims(img, axis=0)  # Add batch dim

def draw_single_box(image, box, label, score, threshold=0.4):
    """Draw one bounding box + label on the image."""
    if score < threshold:
        return image

    h, w, _ = image.shape
    xmin, ymin, xmax, ymax = box
    xmin, xmax = int(xmin * w), int(xmax * w)
    ymin, ymax = int(ymin * h), int(ymax * h)

    color = (0, 255, 0)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    text = f"{CLASS_NAMES[label]} ({score:.2f})"
    cv2.putText(image, text, (xmin, max(ymin - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def run_inference(model_path, image_path, save_dir):
    print(f"🔍 Running inference on {os.path.basename(image_path)} ...")
    model = tf.keras.models.load_model(model_path, compile=False)

    input_img = load_and_preprocess_image(image_path)
    bbox_pred, class_pred = model.predict(input_img)

    bbox_pred = np.squeeze(bbox_pred)
    class_pred = np.squeeze(class_pred)

    label = np.argmax(class_pred)
    score = np.max(class_pred)

    print(f"Detected: {CLASS_NAMES[label]} ({score:.2f})")
    print("Box (normalized):", bbox_pred)

    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    output = draw_single_box(img_cv, bbox_pred, label, score, threshold=0.3)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved result to {save_path}\n")

if __name__ == "__main__":
    model_path = "custom_mobilenet_detector.keras"
    image_dir = "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/test/images"
    save_dir = "/mnt/d/Code/DenseNet-Project/src/steel_defect_oct_2025/train/B/inference_result"

    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith((".jpg", ".png")):
            run_inference(model_path, os.path.join(image_dir, img_name), save_dir)
