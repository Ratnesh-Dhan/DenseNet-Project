from base64 import encode
import os, io, json
from turtle import width
import tensorflow as tf
from PIL import Image

CLASSES = ["bus", "car", "motorbike", "threewheel", "truck", "van"]
label_map = {name: i+1 for i, name in enumerate(CLASSES)}

def create_tf_example(img_path, json_path):
    # Load image
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded = fid.read()
    image = Image.open(io.BytesIO(encoded))
    width, height = image.size
    filename = os.path.basename(img_path)

    # Load annotation JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    xmins, xmaxs, ymins, ymaxs, classes, classes_text = [],[],[],[],[],[]
    for obj in data.get("objects", []):
        cls = obj["classTitle"]
        if cls not in label_map:
            continue
        (xmin, ymin), (xmax, ymax) = obj["points"]["exterior"]

        # Normalize
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes.append(label_map[cls])
        classes_text.append(cls.encode("utf8"))
    
    feature = {
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode("utf8")])),
        "image/source_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode("utf8")])),
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"jpg"])),
        "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert(images_dir, ann_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    for file in os.listdir(images_dir):
        if not file.lower().endswith((".jpg",".jpeg",".png")): continue
        img_path = os.path.join(images_dir, file)
        ann_path = os.path.join(ann_dir, os.path.splitext(file)[0] + ".json")
        if not os.path.exists(ann_path):
            continue
        tf_example = create_tf_example(img_path, ann_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("✅ Wrote TFRecord:", output_path)

if __name__ == "__main__":
    # Example usage:
    # /mnt/d/DATASETS/vehicle-dataset-for-yolo
    convert("/mnt/d/DATASETS/vehicle-dataset-for-yolo/train/img", "/mnt/d/DATASETS/vehicle-dataset-for-yolo/train/ann", "train.record")
    convert("/mnt/d/DATASETS/vehicle-dataset-for-yolo/valid/img", "/mnt/d/DATASETS/vehicle-dataset-for-yolo/valid/ann", "val.record")