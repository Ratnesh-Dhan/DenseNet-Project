import json
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from model import create_model
from mini_VGG_model import create_better_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns

train_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\16\TRAINING 16"
validation_dir = r"D:\NML ML Works\newCoalByDeepBhaiya\16\VALIDATION"
model_name = "newCNNjune13Epoch_100"
batch_size = 64
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(16, 16),
    batch_size=batch_size,
    class_mode='sparse',
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(16, 16),
    batch_size=batch_size,
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
    filepath=f"../models/EarlyStoppedBest{model_name}.keras",
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
    epochs=100,
    # callbacks=[early_stop, model_checkpoint]
)
# model.save('new_folder.h5')
model.save(f'../models/{model_name}.keras')

with open('../results/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

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