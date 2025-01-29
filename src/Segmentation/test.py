from matplotlib import pyplot as plt

path = "../img/cat1.jpg"
image = plt.imread(path)
plt.imshow(image)
plt.axis("off")
plt.show()