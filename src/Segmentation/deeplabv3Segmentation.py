# -*- coding: utf-8 -*-
# USING EXISTING MODEL FROM KAGGLE NAMED deeplabv3 FOR SEGMENTAION

import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

model_path = "../../TrainedModel/2.tflite"

model = tf.keras.models.