# VID20210601143927-96_jpg.rf.36de73b8200ee94d0bd4679407c9cd40
import os
import matplotlib.pyplot as plt
import json, sys, cv2

def get_obj(id: str, meta):
    for i in meta['classes']:
        if id == i['id']:
            return i
    
root = "../../Datasets/pcbDataset"
image_name = "VID20210601143927-96_jpg.rf.36de73b8200ee94d0bd4679407c9cd40.jpg"
image_path = os.path.join(root, "train", "img", image_name)  # Join root with "img/" and the image name
annotation = os.path.join(root, "train", "ann", image_name+'.json')
meta_path = os.path.join(root, "meta.json")

# Loading image
image = cv2.imread(image_path)
# Loading meta.json file
with open(meta_path, 'r') as met:
    meta = json.load(met)

# Loading annotation file
with open(annotation, 'r') as file:
    data = json.load(file)

for i in data['objects']:
    box = i['points']['exterior']
    # print(box[0][0], " ", box[0][1])
    # print(box[1][0], " ", box[1][1])
    obj = get_obj(i['classId'], meta)
    color_hex = obj['color']
    color_bgr = [int(color_hex[i:i+2], 16) for i in (5, 3, 1)]  # Convert hex to BGR
    cv2.putText(image, obj['title'], (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, color_bgr, 2)
    cv2.rectangle(image, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color_bgr, 2)
    print(color_bgr)

# sys.exit(0)
plt.imshow(image)
plt.show()