import json 

def test_single_json():
    json_path = '../../Datasets/PASCAL VOC 2012/train/ann/2007_000250.jpg.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Print the structure of your JSON
    print("JSON keys:", data.keys())
    if 'objects' in data:
        print("Number of objects:", len(data['objects']))
        for obj in data['objects']:
            print("Object keys:", obj.keys())
            if 'bitmap' in obj:
                print("Bitmap keys:", obj['bitmap'].keys())

if __name__ == '__main__':
    test_single_json()  # Run this instead of main() for testing