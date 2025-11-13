import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

class GraphUtils:
    def __init__(self, save_dir="results"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def plot_loss_accuracy(self, history, optimizer_name):
        plt.figure()
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title(f"Loss ({optimizer_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{optimizer_name}_loss.png"))
        plt.close()

        plt.figure()
        plt.plot(history["train_acc"], label="Train Accuracy")
        plt.plot(history["val_acc"], label="Val Accuracy")
        plt.title(f"Accuracy ({optimizer_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{optimizer_name}_accuracy.png"))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, optimizer_name, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix ({optimizer_name})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(self.save_dir, f"{optimizer_name}_confusion_matrix.png"))
        plt.close()

    def get_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def plot_optimizer_comparison(self, results):
        names = [r["optimizer"] for r in results]
        f1s = [r["f1"] for r in results]
        precs = [r["precision"] for r in results]
        recs = [r["recall"] for r in results]
        accs = [r["accuracy"] for r in results]

        x = np.arange(len(names))
        width = 0.2

        plt.figure(figsize=(10, 6))
        plt.bar(x - 1.5*width, accs, width, label="Accuracy")
        plt.bar(x - 0.5*width, precs, width, label="Precision")
        plt.bar(x + 0.5*width, recs, width, label="Recall")
        plt.bar(x + 1.5*width, f1s, width, label="F1")
        plt.xticks(x, names)
        plt.title("Optimizer Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "optimizer_comparison.png"))
        plt.close()
 ## Usage:
# graphs = GraphUtils("results")
# metrics = graphs.get_metrics(y_true, y_pred)
# graphs.plot_confusion_matrix(y_true, y_pred, "Adam", class_names)
# graphs.plot_loss_accuracy(history, "Adam")
# graphs.plot_optimizer_comparison(results)
