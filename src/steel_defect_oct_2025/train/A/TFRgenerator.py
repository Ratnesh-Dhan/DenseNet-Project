import tensorflow as tf
import xml.etree.ElementTree as ET
import os

# class map
CLASS_MAP = {"scratches":0, "inclusion":1, "class3":2, "class4":3, "class5":4, "class6":5}

def create_tf_example(xml_file, image_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = os.path.join(image_dir, filename)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    height, width, _ = img.shape

    xmins, ymins, xmaxs, ymaxs, classes_text, classes = [], [], [], [], [], []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text) / width
        ymin = float(bndbox.find("ymin").text) / height
        xmax = float(bndbox.find("xmax").text) / width
        ymax = float(bndbox.find("ymax").text) / height

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        classes_text.append(label.encode("utf8"))
        classes.append(CLASS_MAP[label])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode("utf8")])),
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(img).numpy()])),
        "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example
