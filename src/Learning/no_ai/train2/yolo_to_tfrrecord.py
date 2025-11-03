import tensorflow as tf
import os, cv2
from object_detection.utils import dataset_util

# --- CONFIG ---
DATASET_DIR = "dataset"
OUTPUT_DIR = "annotations"
LABELS = ["car", "bus", "bike"]  # <-- Change to your class names

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_tf_example(img_path, label_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    filename = os.path.basename(img_path).encode('utf8')

    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                cls = int(cls)
                x_center, y_center = x * width, y * height
                box_w, box_h = w * width, h * height

                xmin = (x_center - box_w / 2.0) / width
                xmax = (x_center + box_w / 2.0) / width
                ymin = (y_center - box_h / 2.0) / height
                ymax = (y_center + box_h / 2.0) / height

                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                classes_text.append(LABELS[cls].encode('utf8'))
                classes.append(cls + 1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(b'jpg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def convert_to_tfrecord(split):
    writer = tf.io.TFRecordWriter(os.path.join(OUTPUT_DIR, f"{split}.record"))
    img_dir = os.path.join(DATASET_DIR, "images", split)
    label_dir = os.path.join(DATASET_DIR, "labels", split)

    for img_file in os.listdir(img_dir):
        if not img_file.endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
        tf_example = create_tf_example(img_path, label_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"✅ Done creating {split}.record")

convert_to_tfrecord("train")
convert_to_tfrecord("val")
