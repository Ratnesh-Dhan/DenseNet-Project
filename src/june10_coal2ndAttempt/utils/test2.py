import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Original confusion matrix (rows: actual, columns: predicted)
cm = np.array([
    [623, 33,   0,   2,   0],
    [ 73, 486, 51,  39,  11],
    [  0,   1, 1138, 50,  24],
    [ 28,  8, 245, 574,   7],
    [  0,   2, 301, 2,  508]
])

# Normalize row-wise (per actual class)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
cm_percent = cm_normalized * 100

# Labels for axes
labels = [0, 1, 2, 3, 4]

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix (%)")
plt.tight_layout()

# Save or show the plot
plt.savefig("confusion_matrix_percent_this_one.png")
plt.show()
