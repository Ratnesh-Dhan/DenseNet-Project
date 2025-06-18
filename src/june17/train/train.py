import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import create_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Paths
train_dir = r'D:\NML ML Works\new square corrosion dataset\dataset\Train'  # Your single folder with subfolders per class

# Parameters
IMAGE_SIZE = (30, 30)
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42  # ensures reproducible splits

# Data generator with split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    seed=SEED
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    seed=SEED
)

# Model and callbacks
model = create_model()
model.summary()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
]

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save("my_fully_trained_model.keras")

# Create a directory to store results
os.makedirs("results", exist_ok=True)

# 1. Plot training/validation accuracy & loss
def plot_history(history):
    plt.figure(figsize=(10, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_history.png")
    plt.close()

plot_history(history)

# 2. Predict on validation set
val_generator.reset()  # reset so predictions match order
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# 3. Confusion matrix (normalized)
cm = confusion_matrix(y_true, y_pred, normalize='true')
labels = list(val_generator.class_indices.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig("results/confusion_matrix.png")
plt.close()

# 4. Classification report
report = classification_report(y_true, y_pred, target_names=labels)
with open("results/classification_report.txt", "w") as f:
    f.write(report)

# 5. Print accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy:", accuracy)
