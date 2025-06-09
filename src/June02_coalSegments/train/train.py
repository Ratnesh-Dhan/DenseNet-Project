import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

train_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\TRAINING"
validation_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\VALIDATION"

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(31, 31),
    batch_size=32,
    class_mode='sparse',
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(31, 31),
    batch_size=32,
    class_mode='sparse',
    shuffle=False # Important for prediction-evaluation match
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience = 5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath="../models/EarlyStoppedBest09June.keras",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only=True,
    mode='auto'
)

model = create_model()
# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=50,
    callbacks=[early_stop, model_checkpoint]
)
# model.save('new_folder.h5')
model.save('../models/modelJUNE09.keras')

# Plot the training and validation accuracy and loss
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.plot(history.history['loss'], label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("./results/09.png", bbox_inches="tight")
plt.show()


# Evaluate the model on the validation dataset
validation_loss, validation_acc = model.evaluate(validation_generator)
print('Validation accuracy:', validation_acc)

# Use the model to make predictions on the validation dataset
predictions = model.predict(validation_generator, steps=len(validation_generator), verbose=1)

# Convert the predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Print the classification report and confusion matrix
# print('Classification Report:')
# print(classification_report(validation_generator.classes, predicted_classes))
# print('Confusion Matrix:')
# print(confusion_matrix(validation_generator.classes, predicted_classes))
true_labels = validation_generator.classes[:len(predicted_classes)]
# Save classification report and confusion matrix to a text file
with open('./results/09_metrics.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report(true_labels, predicted_classes))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(true_labels, predicted_classes)))
