import tensorflow as tf
import json, os, numpy as np, glob, base64, io, zlib, sys
from PIL import Image

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def decode_bitmap(bitmap_data, origin, height, width):
    decoded_data = base64.b64decode(bitmap_data)
    try:
        # Decompress and convert to image
        decompressed_data = zlib.decompress(decoded_data)
        bitmap_img = Image.open(io.BytesIO(decompressed_data))
        bitmap_array = np.array(bitmap_img.convert('L'))
    except Exception as e:
        print(f"Error in decode bitmap: {e}")
        return np.zeros((height, width), dtype=np.uint8)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y = origin
    h, w = bitmap_array.shape[:2]
    mask[y:y+h, x:x+w] = bitmap_array > 0
    return mask

def create_tfrecord(image_paths, output_path, metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    class_id_to_index = {cls['id']: i for i, cls in enumerate(metadata['classes'])}
    i = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for image_path in image_paths:
            i = i+1
            print(f"{i} out of {len(image_paths)}")
            image_name = os.path.basename(image_path)
            # img = Image.open(image_path)
            with tf.io.gfile.GFile(image_path, 'rb') as f:
                image_data = f.read()

            ann_path = os.path.join(
                os.path.dirname(os.path.dirname(image_path)),
                'ann',
                f'{image_name}.json'
            )
            with open(ann_path, 'r') as f:
                annotation = json.load(f)

            height, width = annotation['size']['height'], annotation['size']['width']
            mask = np.zeros((height, width), dtype=np.uint8)

            for obj in annotation['objects']:
                class_idx = class_id_to_index.get(obj['classId'], 0)
                if obj['geometryType'] == 'bitmap':
                    obj_mask = decode_bitmap(
                        obj['bitmap']['data'],
                        obj['bitmap']['origin'],
                        height,
                        width
                    )
                    mask[obj_mask > 0] = class_idx + 1

            mask_pil = Image.fromarray(mask)
            mask_buffer = io.BytesIO()
            mask_pil.save(mask_buffer, format='PNG')
            mask_encoded = mask_buffer.getvalue()

            feature = {
                'image/encoded': _bytes_feature(image_data),
                'image/filename': _bytes_feature(image_name.encode('utf8')),
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/segmentation/class/encoded': _bytes_feature(mask_encoded)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def main():
    # Set paths
    dataset_dir = '../../Datasets/PASCAL VOC 2012/train/'
    image_dir = os.path.join(dataset_dir, 'img')
    metadata_path = os.path.join(dataset_dir, '../meta.json')
    print(metadata_path)
    
    # Get all image paths
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    # Split into train/val sets (adjust split ratio as needed)
    np.random.shuffle(image_paths)
    split_idx = int(len(image_paths) * 0.8)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(dataset_dir, 'tfrecords'), exist_ok=True)
    
    # Create TFRecords
    create_tfrecord(
        train_paths,
        os.path.join(dataset_dir, 'tfrecords', 'train.tfrecord'),
        metadata_path
    )
    create_tfrecord(
        val_paths,
        os.path.join(dataset_dir, 'tfrecords', 'val.tfrecord'),
        metadata_path
    )

if __name__ == '__main__':
    main()

