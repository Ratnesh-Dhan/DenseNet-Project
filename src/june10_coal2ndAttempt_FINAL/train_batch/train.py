import json, os, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
from dataset_loader import load_dataset
import tensorflow as tf


train_generator, validation_generator = load_dataset(batch_size = 64)

optimizers = {
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.001),
    "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=0.01),
    "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    "Adadelta": tf.keras.optimizers.Adadelta(learning_rate=1.0),
    "Nadam": tf.keras.optimizers.Nadam(learning_rate=0.001)
}

early_stop = EarlyStopping(
    monitor='val_loss',
    patience = 5,
    restore_best_weights=True,
    verbose=1
)

summary = {}

# Starting the loop 
for name , optimizer in optimizers.items():

    print(f"\n🔧 Training with {name} optimizer\n")

    os.makedirs(f"./results/{name}", exist_ok=True)
    os.makedirs(f"./models/{name}", exist_ok=True)

    model_checkpoint = ModelCheckpoint(
        filepath=f"./models/{name}/checkpoint_best_weights.keras",
        monitor = 'val_loss',
        verbose = 1,
        save_best_only=True,
        mode='auto'
    )

    model = create_model(optimizer)

    # Training the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        epochs=100,
        callbacks=[early_stop, model_checkpoint]
    )
    last_epoch = len(history.history['loss'])
    model.save(f'./models/{name}/{name}_with_epoch_{last_epoch}.keras')
    # Also save a copy of the best model (early stopped) with epoch info
    best_epoch = np.argmin(history.history['val_loss']) + 1
    model.save(f'./models/{name}/{name}_earlystopped_best_epoch{best_epoch}.keras')

    with open(f'./results/{name}/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)

    # Save training history to CSV
    pd.DataFrame(history.history).to_csv(f'./results/{name}/training_history.csv', index=False)

    # Plot Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/{name}/accuracy_vs_epochs.png', bbox_inches="tight")
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{name} - Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/{name}/loss_vs_epochs.png', bbox_inches="tight")
    plt.close()

    # Evaluate
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f"✅ {name} Validation Accuracy: {val_acc:.4f}")

    # Save prediction metrics
    predictions = model.predict(validation_generator, steps=len(validation_generator), verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_labels = validation_generator.classes[:len(predicted_classes)]

    report = classification_report(true_labels, predicted_classes, digits=4)
    cm = confusion_matrix(true_labels, predicted_classes)

    # Save raw classification report and confusion matrix
    with open(f'./results/{name}/classification_report.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    # Plot Confusion Matrix (Counts)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_generator.class_indices,
                yticklabels=train_generator.class_indices)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Counts)')
    plt.tight_layout()
    plt.savefig(f'./results/{name}/confusion_matrix_counts.png')
    plt.close()

    # Confusion Matrix Percent
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    np.set_printoptions(precision=2)

    with open(f'./results/{name}/confusion_matrix_percent.txt', 'w') as f:
        f.write(str(cm_percent))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=train_generator.class_indices,
                yticklabels=train_generator.class_indices)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(f'./results/{name}/confusion_matrix_percent.png')
    plt.close()

    # Save summary result
    summary[name] = {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss)
    }

# Save final summary JSON
with open("./results/summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# Print comparison table
print("\n📊 Optimizer Comparison Summary")
print("--------------------------------")
for opt_name, res in summary.items():
    print(f"{opt_name:<10}  | Accuracy: {res['val_accuracy']:.4f} | Loss: {res['val_loss']:.4f}")