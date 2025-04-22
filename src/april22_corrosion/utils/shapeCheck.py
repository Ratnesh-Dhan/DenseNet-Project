import os

base_path = r"C:\Users\NDT Lab\Pictures\dataset\archive\corrosion detect"
# Get all files in the directory
files = os.listdir(base_path)

# Print each file
for file in files:
    print(file)
