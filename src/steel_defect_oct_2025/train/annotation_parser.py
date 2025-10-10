import os
import xml.etree.ElementTree as ET

def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    boxes = []
    labels = []
    
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # Normalize [0,1] because models expect relative coords
        boxes.append([xmin/width, ymin/height, xmax/width, ymax/height])
        labels.append(label)
    
    return filename, boxes, labels
