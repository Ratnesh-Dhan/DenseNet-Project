import os

base_path = r"D:\NML ML Works\newCoalByDeepBhaiya"
files = os.listdir(os.path.join(base_path, "16/TRAINING 16"))

target_path = os.path.join(base_path, "16/VALIDATION")
if not os.path.exists(target_path):
    os.makedirs(target_path)

for directory in files:
    folder = os.path.join(target_path, directory)
    if not os.path.exists(folder):
        os.makedirs(folder)