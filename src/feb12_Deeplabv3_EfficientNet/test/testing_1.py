import numpy as np
from shapely.geometry import Polygon, Point
import json
from matplotlib import pyplot as plt

def parse_polygon(coordinates):
    """Convert polygon coordinates to Shapely Polygon object."""
    return Polygon(coordinates)

def create_mask_from_polygons(polygons, height, width):
    """Create a binary mask from a list of polygons."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create a grid of points
    y, x = np.mgrid[:height, :width]
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    
    # Check each point if it's inside any polygon
    for polygon in polygons:
        for idx, point in enumerate(points):
            if polygon.contains(Point(point)):
                mask[point[1], point[0]] = 1
    
    return mask

def json_to_mask(json_data, height, width):
    """
    Convert JSON annotation data to binary mask.
    
    Parameters:
    json_data (str or dict): JSON data containing polygon coordinates
    height (int): Height of the output mask
    width (int): Width of the output mask
    
    Returns:
    numpy.ndarray: Binary mask of shape (height, width)
    """
    # Parse JSON if string is provided
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Extract polygons from JSON
    polygons = []
    for annotation in data['annotations']:
        if 'coordinates' in annotation:
            polygon = parse_polygon(annotation['coordinates'])
            polygons.append(polygon)
    
    # Create mask from polygons
    mask = create_mask_from_polygons(polygons, height, width)
    
    return mask

# Example usage
# example_json = {
#     "annotations": [
#         {
#             "coordinates": [[10, 10], [10, 20], [20, 20], [20, 10]]
#         }
#     ]
# }

with open('../../../Datasets/PASCAL VOC 2012/train/ann/2011_000646.jpg.json', 'r') as file:
    example_json = json.load(file)

# Create a 30x30 mask
mask = json_to_mask(example_json, height=30, width=30)
plt.imshow(mask)
plt.axis('off')
plt.show()