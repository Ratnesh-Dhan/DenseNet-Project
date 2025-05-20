# train_unet.py
from model import build_unet
from dataset_loader import CorrosionDataset
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + smooth) / (denominator + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
) 

base_location = r'D:\NML ML Works\kaggle_semantic_segmentation_CORROSION_dataset'
train_gen = CorrosionDataset(os.path.join(base_location, "train/images"), os.path.join(base_location, "train/masks"))
val_gen = CorrosionDataset(os.path.join(base_location, "validate/images"), os.path.join(base_location, "validate/masks"))

model = build_unet(input_shape=(256, 256, 3))
model.compile(optimizer="adam", loss=bce_dice_loss, metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[early_stopping])
model.save("../models/unet_resnet50_corrosion.h5")
