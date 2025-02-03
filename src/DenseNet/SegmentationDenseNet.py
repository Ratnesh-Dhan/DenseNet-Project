# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:25:22 2025
 USING MY OWN TRAINED CUSTOM MODEL
@author: NDT Lab
"""

import tensorflow as tf
from TrainingDenseNet import DenseLayer, DenseBlock, TransitionLayer, DenseNet

with tf.keras.utils.custom_object_scope({'DenseLayer': DenseLayer, 'DenseBlock': DenseBlock, 'TransitionLayer': TransitionLayer, 'DenseNet': DenseNet}):
    model = tf.keras.models.load_model('../../TrainedModel/TestDensenet_cifar10.h5')

if model:
    model.summary()