import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# === Your classes ===
# CLASS_MAP = {
#     "crazing": 0,
#     "inclusion": 1,
#     "patches": 2,
#     "pitted_surface": 3,
#     "rolled-in_scale": 4,
#     "scratches": 5
# }
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def load_and_preprocess_image(image_path, target_size=(200, 200)):
    """Load and normalize image for inference."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    return tf.expand_dims(img, axis=0)  # (1, 200, 200, 3)

def draw_boxes(image, boxes, labels, scores, threshold=0.4):
    """Draw bounding boxes and labels."""
    h, w, _ = image.shape
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)

        color = (0, 255, 0)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        text = f"{CLASS_NAMES[label]}: {score:.2f}"
        cv2.putText(image, text, (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def run_inference(model_path, image_path, save_output):
    """Run object detection on one image."""
    print("🚀 Loading model...")
    model = tf.keras.models.load_model(model_path)

    input_img = load_and_preprocess_image(image_path)
    preds = model.predict(input_img)
    print("Printing predictoin : ",preds)
    # === Expected output shape: (num_detections, 4 + num_classes)
    bbox_pred, class_pred = preds  # unpack list returned by model

    # Convert tensors to numpy arrays
    bbox_pred = np.array(bbox_pred)
    class_pred = np.array(class_pred)

    # Remove batch dimension safely
    bbox_pred = np.squeeze(bbox_pred, axis=0)
    class_pred = np.squeeze(class_pred, axis=0)

    # Handle single or multiple detections
    if bbox_pred.ndim == 1:  # only one box
        bbox_pred = np.expand_dims(bbox_pred, axis=0)
        class_pred = np.expand_dims(class_pred, axis=0)

    # Get class labels & scores for each detection
    labels = np.argmax(class_pred, axis=1)
    scores = np.max(class_pred, axis=1)

    # Print to debug
    print(f"Detected {len(labels)} objects")
    print("Boxes:", bbox_pred)
    print("Labels:", labels)
    print("Scores:", scores)

    # Draw boxes
    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    output = draw_boxes(img_cv, bbox_pred, labels, scores, threshold=0.4)
    save_path = os.path.join(save_output, os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved detected image at: {save_path}")


if __name__ == "__main__":
    model_name="custom_mobilenet_detector.keras"

    # for multiple 
    images = "/mnt/d/Code/DenseNet-Project/Datasets/NEU-DET/test/images"
    # images = "../../../../Datasets/NEU-DET/test/images"
    save_path = "/mnt/d/Code/DenseNet-Project/src/steel_defect_oct_2025/train/B/inference_result"
    os.makedirs(save_path, exist_ok=True)

    images_all = os.listdir(images)
    for img in images_all:
        img_path = os.path.join(images, img)
        run_inference(model_name, img_path, save_path)

    sys.exit(0)
    image_path="../../../../Datasets/NEU-DET/validation/images/inclusion/inclusion_241.jpg"
    # run_inference(args.model, args.image, save_output=args.save)
    run_inference(model_name, image_path, "../../results")

