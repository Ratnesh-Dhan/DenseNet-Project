import pandas as pd
import os
import numpy as np

all_files = os.listdir("/mnt/d/Codes/DenseNet-Project/src/FringMarch2026/testRA")

with open("Original_csv_Results.txt", 'w') as f:
    for file in all_files:
        csv_file = pd.read_csv(os.path.join("/mnt/d/Codes/DenseNet-Project/src/FringMarch2026/testRA", file))
        numpy_array  = csv_file.to_numpy()
        numpy_array = (numpy_array - numpy_array.mean()) / numpy_array.std()
        print(f"\nFor {file}")
        f.write(f"\nFor {file}\n")
        mean = np.mean(numpy_array)
        Ra = np.mean(np.abs(numpy_array-mean))
        Rz = np.max(numpy_array) -np.min(numpy_array)

        print(f"Ra value : {Ra}")
        print(f"Rz value : {Rz}\n")
        f.write(f"Ra value : {Ra}\n")
        f.write(f"Rz value : {Rz}\n\n")
    f.close()