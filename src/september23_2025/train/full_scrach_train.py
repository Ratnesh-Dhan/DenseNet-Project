from random import shuffle
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import LVGG16_16x16
import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import absl.logging
import tensorflow as tf

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

train_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/TRAINING 16"
validation_dir = r"/mnt/d/NML ML Works/newCoalByDeepBhaiya/16/VALIDATION"

model_name = "Septmber23"
# batch_size = 64 # for sparse categorical
batch_size=64
img_size=(16,16)

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
)

val_ds = keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
)

normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x,y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# train_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
# )
# validation_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
# )

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(16,16),
#     batch_size=batch_size,
#     # class_mode='sparse', # for sparse categorical entropy
#     class_mode='categorical'
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(16,16),
#     batch_size=batch_size,
#     # class_mode='sparse',
#     class_mode='categorical',
#     shuffle=False
# )

early_stop=EarlyStopping(
    monitor='val_loss',
    patience = 5,
    restore_best_weights=True,
    # verbose=1
)

model_checkpoint=ModelCheckpoint(
    filepath=f"../models/EarlyStoppedBest{model_name}.keras",
    monitor = 'val_loss',
    # verbose = 1,
    save_best_only=True,
    mode='auto'
)

model = LVGG16_16x16(num_classes=5)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stop, model_checkpoint]
)
# history=model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator),
#     epochs=30,
#     callbacks=[early_stop, model_checkpoint]
# )

model.save(f'../models/{model_name}.keras')

with open('../results/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.xlim(0)       # Start x-axis at 0
plt.ylim(0)       # Start y-axis at 0

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.xlim(0)       # Start x-axis at 0
plt.ylim(0)       # Start y-axis at 0

plt.tight_layout()
plt.savefig(f"../results/{model_name}.png", bbox_inches="tight")
plt.show()

# Evaluate the model on the validation dataset
validation_loss, validation_acc = model.evaluate(validation_generator)
print('Validation accuracy:', validation_acc)

# Use the model to make predictions on the validation dataset
predictions = model.predict(validation_generator, steps=len(validation_generator), verbose=1)

# Convert the predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

true_labels = validation_generator.classes[:len(predicted_classes)]
# Save classification report and confusion matrix to a text file
with open(f'../results/{model_name}_metrics.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(true_labels, predicted_classes))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(true_labels, predicted_classes)))

cm = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices, yticklabels=train_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f'../results/{model_name}_confusion_matrix.png')
plt.close()


# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)

# Normalize the confusion matrix row-wise (percentage)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Save classification report and confusion matrix to a text file
with open(f'../results/{model_name}_metrics2.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(true_labels, predicted_classes))
    f.write("\n\nConfusion Matrix (%):\n")
    np.set_printoptions(precision=2)
    f.write(str(cm_percent))

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=train_generator.class_indices,
            yticklabels=train_generator.class_indices)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (%)')
plt.tight_layout()
plt.savefig(f'../results/{model_name}_confusion_matrix_percent.png')
plt.close()