import pandas as pd
import numpy as np
import os

excel_dir = "/mnt/d/Codes/OLE-sai_kamal/excel_files"
save_dir = "/mnt/d/Codes/OLE-sai_kamal/heightmaps"
os.makedirs(save_dir, exist_ok=True)

for file in os.listdir(excel_dir):
    if file.endswith(".csv"):
        arr = pd.read_csv(os.path.join(excel_dir, file), header=None).values
        np.save(os.path.join(save_dir, file.replace(".csv",".npy")), arr)