import matplotlib.pyplot as plt

image_path = "../images/testThispng.png"

image = plt.imread(image_path)
plt.imshow(image)
plt.show()