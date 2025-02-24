import tensorflow as tf
import json, os, numpy as np, glob, base64, io, sys
from PIL import Image

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def decode_bitmap(bitmap_data, origin, height, width):
    """
    Decode base64 bitmap data and create a binary mask
    
    Args:
        bitmap_data: Base64 encoded bitmap data
        origin: [x, y] coordinates of bitmap origin
        height: Image height
        width: Image width
    Returns:
        numpy array of shape (height, width) with binary mask
    """
    # Decode base64 data
    decoded_data = base64.b64decode(bitmap_data)
    
    # Convert to image
    bitmap_img = Image.open(io.BytesIO(decoded_data))
    bitmap_array = np.array(bitmap_img)
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Place bitmap at correct position
    x, y = origin
    h, w = bitmap_array.shape[:2]
    mask[y:y+h, x:x+w] = bitmap_array[:, :, 3] > 0  # Use alpha channel for mask
    
    return mask

def create_tfrecord(image_paths, output_path, metadata_path):
    """
    Convert images and annotations to TFRecord format.
    
    Args:
        image_paths: List of paths to images
        output_path: Path to save TFRecord file
        metadata_path: Path to metadata.json
    """
    # Read class metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create class id to index mapping
    class_id_to_index = {}
    for i, cls in enumerate(metadata['classes']):
        class_id_to_index[cls['id']] = i
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_path) as writer:
        for image_path in image_paths:
            # Get image name
            image_name = os.path.basename(image_path)
            
            # Read image
            img = Image.open(image_path)
            with tf.io.gfile.GFile(image_path, 'rb') as f:
                image_data = f.read()
            
            # Read annotation file
            ann_path = os.path.join(
                os.path.dirname(os.path.dirname(image_path)),
                'ann',
                f'{image_name}.json'
            )
            
            with open(ann_path, 'r') as f:
                annotation = json.load(f)
            
            # Create segmentation mask
            height = annotation['size']['height']
            width = annotation['size']['width']
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Fill mask for each object
            for obj in annotation['objects']:
                class_idx = class_id_to_index[obj['classId']]
                if obj['geometryType'] == 'bitmap':
                    obj_mask = decode_bitmap(
                        obj['bitmap']['data'],
                        obj['bitmap']['origin'],
                        height,
                        width
                    )
                    # Add class index to mask (add 1 to reserve 0 for background)
                    mask[obj_mask > 0] = class_idx + 1
            
            # Encode mask as PNG
            mask_pil = Image.fromarray(mask)
            mask_buffer = io.BytesIO()
            mask_pil.save(mask_buffer, format='PNG')
            mask_encoded = mask_buffer.getvalue()
            
            # Create feature dictionary
            feature = {
                'image/encoded': _bytes_feature(image_data),
                'image/filename': _bytes_feature(image_name.encode('utf8')),
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/segmentation/class/encoded': _bytes_feature(mask_encoded)
            }
            
            # Create example and write to TFRecord
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def main():
    # Set paths
    dataset_dir = '../../Datasets/testDataset/'
    image_dir = os.path.join(dataset_dir, 'img')
    metadata_path = os.path.join(dataset_dir, 'meta.json')
    
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