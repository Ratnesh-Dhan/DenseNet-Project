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
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(31, 31),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

model = create_model()
# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=10
)
model.save('model.keras')

# Plot the training and validation accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the validation dataset
validation_loss, validation_acc = model.evaluate(validation_generator)
print('Validation accuracy:', validation_acc)

# Use the model to make predictions on the validation dataset
predictions = model.predict(validation_generator)

# Convert the predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Print the classification report and confusion matrix
print('Classification Report:')
print(classification_report(validation_generator.classes, predicted_classes))
print('Confusion Matrix:')
print(confusion_matrix(validation_generator.classes, predicted_classes))
