from matplotlib import pyplot as plt
import os, numpy as np

# file_path = r'D:\\'
# image_path = os.path.join(file_path, "apple.webp")

# ary = [[1,2,3,4,5,6,7,8,9,10],
#        [10,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10],
#        [1,2,3,4,5,6,7,8,9,10]]

ary = np.zeros((11,11))
# ary = [[0 for _ in range(10)] for _ in range(10)]

for i in range(11):
    for j in range(11):
        if i == 5 and j == 5:
            ary[i][j] = 5
        elif i == j:
            ary[i][j] = 1
        elif i + j == 10:
            ary[i][j] = 1
        else:
            ary[i][j] = 0

ary = np.array(ary)
plt.imshow(ary)
plt.show()

# print(ary[2:5, 2:5])






