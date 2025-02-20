import numpy as np
import base64
import json
import matplotlib.pyplot as plt

def decode_base64_to_mask(data, origin, height, width):
    """Decode base64 bitmap data to binary mask."""
    # Decode base64 string to bytes
    decoded_data = base64.b64decode(data)
    
    # Convert bytes to numpy array
    bitmap = np.frombuffer(decoded_data, dtype=np.uint8)
    
    # Create full-size mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Place the bitmap in the correct position
    x, y = origin
    mask[y:, x:] = bitmap.reshape(-1, 1)
    
    return mask

def load_and_show_masks(json_path):
    """Load JSON file and show masks for each class."""
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    height = data['size']['height']
    width = data['size']['width']
    
    # Create figure with subplots for each class
    unique_classes = set(obj['classTitle'] for obj in data['objects'])
    n_classes = len(unique_classes)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    
    # Process each class
    for ax, class_name in zip(axes, unique_classes):
        # Create mask for this class
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Combine all objects of this class
        for obj in data['objects']:
            if obj['classTitle'] == class_name:
                obj_mask = decode_base64_to_mask(
                    obj['bitmap']['data'],
                    obj['bitmap']['origin'],
                    height, width
                )
                mask = np.maximum(mask, obj_mask)
        
        # Show mask
        ax.imshow(mask, cmap='gray')
        ax.set_title(f'Mask for {class_name}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# with open('../../../Datasets/PASCAL VOC 2012/train/ann/2011_000646.jpg.json', 'r') as file:
#     example_json = json.load(file)

# Load and show masks
json_path = '../../../Datasets/PASCAL VOC 2012/train/ann/2011_000646.jpg.json' 
load_and_show_masks(json_path)