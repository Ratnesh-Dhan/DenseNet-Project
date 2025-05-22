import os
import tensorflow as tf
from model import unet_model
from new_transfer_model import build_unet_with_resnet50
import matplotlib.pyplot as plt
from data_loader_old import get_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# model = unet_model()
model = build_unet_with_resnet50()

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# model.compile(optimizer='adam',
#               loss=bce_dice_loss,
#               metrics=[dice_loss])

model.summary()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

base_dir = r"/home/zumbie/Codes/NML/DenseNet-Project/Datasets/kaggle_semantic_segmentation_CORROSION_dataset"
# base_dir = r"D:\NML ML Works\kaggle_semantic_segmentation_CORROSION_dataset"
train_ds = get_dataset(os.path.join(base_dir, "train/images"), os.path.join(base_dir, "train/masks"), batch_size=8)
val_ds = get_dataset(os.path.join(base_dir, "validate/images"), os.path.join(base_dir, "validate/masks"), batch_size=8)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model_transferLearning_test.h5', monitor='val_loss', save_best_only=True)
]
#  callbacks=[tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True)], 

history = model.fit(train_ds, 
                    validation_data=val_ds, 
                    epochs=80,
                    callbacks=callbacks
                    )
model.save("final_model_transferLearning.h5")
print('\nModel saved successfully..!\n')
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy (optional if using binary accuracy metric)
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history)
