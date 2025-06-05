import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from model import create_model

train_dir = r"D:\NML ML Works\TRAINING-20250602T050431Z-1-001\working dataset\train"
validation_dir = r"D:\NML ML Works\TRAINING-20250602T050431Z-1-001\working dataset\validation"

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

model = create_model()
# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=10
)
model.save('new_folder.h5')
model.save('model.keras')

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
print(classification_report(true_labels, predicted_classes))
print(confusion_matrix(true_labels, predicted_classes))
