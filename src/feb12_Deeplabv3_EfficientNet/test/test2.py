import json 
import base64
from PIL import Image
import io
import zlib

def test_single_json():
    json_path = '../../Datasets/PASCAL VOC 2012/train/ann/2007_000250.jpg.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("Testing JSON file:", json_path)
    print("Image size:", data['size'])
    
    if 'objects' in data:
        print("\nNumber of objects:", len(data['objects']))
        for i, obj in enumerate(data['objects']):
            print(f"\nObject {i+1}:")
            print("Class ID:", obj['classId'])
            print("Class Title:", obj['classTitle'])
            print("Origin:", obj['bitmap']['origin'])
            
            try:
                # Decode and decompress bitmap data
                bitmap_data = base64.b64decode(obj['bitmap']['data'])
                decompressed_data = zlib.decompress(bitmap_data)
                
                print(f"Decoded and decompressed data length: {len(decompressed_data)}")
                
                # Try to open as image
                bitmap_image = Image.open(io.BytesIO(decompressed_data))
                print("Successfully opened bitmap image")
                print("Image mode:", bitmap_image.mode)
                print("Image size:", bitmap_image.size)
                
            except Exception as e:
                print("Error:", str(e))

if __name__ == '__main__':
    test_single_json()