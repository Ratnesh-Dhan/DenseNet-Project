import matplotlib.pyplot as plt
import os, datetime

def plot_history(history):
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")

    # Accuracy
    if 'accuracy' in history.history:
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title("Accuracy")

    save_path = "../Graphs"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"Corrosion_segmentation_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.show()