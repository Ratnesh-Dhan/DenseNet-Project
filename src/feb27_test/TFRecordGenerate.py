import os, json, tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from tqdm import tqdm

DATASET_DIR = '../../Datasets/pcbDataset'
TFRECORD_DIR = 'tfrecord'

SPLITS = ['train', 'validation', 'test']

# Function to crate a tf.train.Example
def create_tf_example(img_path, ann_path):
    with open(ann_path, 'r') as f:
        data = json.load(f)
    img = Image.open(img_path)
    width, height = img.size

    filename = os.path.basename(img_path).encode('utf8')
    encoded_image_data = tf.io.gfile.GFile(img_path, 'rb').read()
    image_format = b'jpeg' if img_path.lower().endswith('.jpg') else b'png'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for obj in data['objects']:
        class_id = obj['classId']
        bbox = obj['points']['exterior']
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]
        xmins.append(xmin/width)
        xmaxs.append(xmax/width)
        ymins.append(ymin/height)
        ymaxs.append(ymax/height)
        classes_text.append(obj['classTitle'].encode('utf8'))
        classes.append(class_id)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Generate TFRecord for each dataset split
def generate_tfrecord(split_name):
    image_dir = os.path.join(DATASET_DIR, split_name, "img")
    ann_dir = os.path.join(DATASET_DIR, split_name, "ann")
    tfrecord_path = os.path.join(TFRECORD_DIR, f'{split_name}.tfrecord')

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))] # Newly added for TQDM

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for filename in tqdm(image_files, desc=f'Processing {split_name} set'):
        # for filename in os.listdir(image_dir):
        #     if not filename.lower().endswidth(('.jpg', '.png')):
        #         continue

            img_path = os.path.join(image_dir, filename)
            ann_path = os.path.join(ann_dir, f'{filename}.json')

            if not os.path.exists(ann_path):
                print(f'Annotation not found for {filename}, so skipping skibidi.')
                continue
            tf_example = create_tf_example(img_path, ann_path)
            writer.write(tf_example.SerializeToString())
        print(f'TFRecord for {split_name} created at {tfrecord_path}')

if __name__ == "__main__":
    os.makedirs(TFRECORD_DIR, exist_ok=True)
    for split in SPLITS:
        generate_tfrecord(split)