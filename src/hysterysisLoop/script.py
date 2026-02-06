# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# df = pd.read_excel("/mnt/d/DATASETS/hysterysis.xlsx")

# x = df["x"].values
# y = df["y"].values

# plt.plot(x, y, "-o", markersize=3)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Hysteresis Data")
# plt.grid(True)
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
df = pd.read_excel("/mnt/d/DATASETS/hysterysis.xlsx")

x = df["x"].to_numpy()
y = df["y"].to_numpy()

# plot (sanity check)
plt.plot(x, y, "-", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")

# close the loop if needed
if x[0] != x[-1] or y[0] != y[-1]:
    x = np.append(x, x[0])
    y = np.append(y, y[0])

# area (shoelace / Green's theorem)
area = 0.5 * abs(
    np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])
)

print("Hysteresis area:", area)


plt.title(f"Hysteresis Loop (Area: {area:.2f})")
plt.axis("auto")
plt.grid(True)
plt.show()

