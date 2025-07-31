import sys
import os

folder_path = r"D:\NML ML Works\newCoalByDeepBhaiya\16\VALIDATION"
move_path = r"D:\NML ML Works\newCoalByDeepBhaiya\16\TRAINING 16"

files = os.listdir(move_path)
files2 = os.listdir(folder_path)

for file in files:
    source = os.path.join(folder_path, file)
    destination = os.path.join(move_path, file)
    images = os.listdir(source)
    print(source)
    for i in images:
        os.rename(os.path.join(source, i), os.path.join(destination, i))
    # # Check if file exists in destination before moving
    # if not os.path.exists(destination):
    #     try:
    #         os.rename(source, destination)
    #         print(f"Moved {file} to training folder")
    #     except Exception as e:
    #         print(f"Error moving {file}: {str(e)}")
    # else:
    #     print(f"File {file} already exists in training folder")




sys.exit(0)
import matplotlib.pyplot as plt
import os

image_path = r"D:\NML ML Works\Deep bhaiya\TESTING2-20250611T124336Z-1-001\TESTING2"
image = plt.imread(os.path.join(image_path, "014.jpg"))
plt.imshow(image)
plt.show()
print(image.shape)