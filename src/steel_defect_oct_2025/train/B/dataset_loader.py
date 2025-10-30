# dataset_loader.py
import tensorflow as tf
import xml.etree.ElementTree as ET
import os

CLASS_MAP = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}

def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)

    boxes, labels = [], []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text) / width
        ymin = int(bndbox.find("ymin").text) / height
        xmax = int(bndbox.find("xmax").text) / width
        ymax = int(bndbox.find("ymax").text) / height
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_MAP[label])
    return filename, boxes, labels

def load_image_and_label(xml_path, img_dir):
    """
    This version is wrapped with tf.py_function to work inside Dataset.map()
    """
    def _py_parse(xml_path_tensor):
        xml_path_str = xml_path_tensor.numpy().decode("utf-8") if isinstance(xml_path_tensor.numpy(), bytes) \
                    else xml_path_tensor.numpy().item()
        filename, boxes, labels = parse_voc_annotation(xml_path_str)

        crop_name = filename.split("_")[0]

        if filename.endswith(".jpg"):
            img_path = os.path.join(img_dir, filename)
        else:
            img_path = os.path.join(img_dir, f"{filename}.jpg")

        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (200, 200)) / 255.0

        return img.numpy(), boxes[0], labels[0]

    img, bbox, label = tf.py_function(
        # func=lambda x: _py_parse(x),
        func=_py_parse,
        inp=[xml_path],
        Tout=[tf.float32, tf.float32, tf.int32]
    )

    # Reshape / set shapes for TensorFlow graph
    img.set_shape((200, 200, 3))
    bbox.set_shape((4,))
    label.set_shape(())

    return img, {"bbox": bbox, "class": label}


# def load_image_and_label_old(xml_file, img_dir):
#     filename, boxes, labels = parse_voc_annotation(xml_file)
#     crop_name = filename.split("_")[0]
#     # if crop_name == "pitted":
#     #     insider_folder_name = "pitted_surface"
#     # elif crop_name == "rolled-in":
#     #     insider_folder_name = "rolled-in_scale"
#     # else:
#     #     insider_folder_name = crop_name
#     print("filename : ", filename)
#     if filename.endswith(".jpg"):
#         img_path = os.path.join(img_dir, filename)
#     else:
#         print(filename)
#         # img_path = os.path.join(img_dir, insider_folder_name, f'{filename}.jpg')
#         img_path = os.path.join(img_dir, f'{filename}.jpg')

#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, (200,200)) / 255.0

#     # For simplicity, take only the first object per image
#     return img, {"bbox": boxes[0], "class": labels[0]}

