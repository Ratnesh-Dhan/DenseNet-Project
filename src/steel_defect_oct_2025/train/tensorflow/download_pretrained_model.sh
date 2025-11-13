#!/bin/bash

# Create directory for pretrained models
mkdir -p pretrained_models
cd pretrained_models

# Download EfficientDet D0 (recommended - good balance of speed and accuracy)
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
tar -xzf efficientdet_d0_coco17_tpu-32.tar.gz
rm efficientdet_d0_coco17_tpu-32.tar.gz

# Alternative: SSD MobileNet V2 (faster, lower accuracy)
# wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# tar -xzf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

# Alternative: Faster R-CNN ResNet50 (slower, higher accuracy)
# wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
# tar -xzf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz

echo "Pretrained model downloaded!"