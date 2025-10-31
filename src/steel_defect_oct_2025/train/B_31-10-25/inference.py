import numpy as np
import cv2, os
import tensorflow as tf
from tf_dataset_loader import CLASS_NAMES

def run_inference(model_path, image_path, save_dir):
    model = tf.keras.models.load_model(model_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (512, 512))
    image_np = image.numpy().astype(np.float32) / 255.0

    preds = model.predict(tf.expand_dims(image_np, 0))
    boxes = preds["boxes"][0].numpy()
    scores = preds["confidence"][0].numpy()
    classes = preds["classes"][0].numpy().astype(int)

    img_cv = cv2.cvtColor(image_np * 255, cv2.COLOR_RGB2BGR)

    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.4:
            continue
        ymin, xmin, ymax, xmax = box
        (xmin, ymin, xmax, ymax) = (int(xmin*512), int(ymin*512), int(xmax*512), int(ymax*512))
        cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        text = f"{CLASS_NAMES[cls]}: {score:.2f}"
        cv2.putText(img_cv, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, img_cv)
    print("✅ Saved:", out_path)
