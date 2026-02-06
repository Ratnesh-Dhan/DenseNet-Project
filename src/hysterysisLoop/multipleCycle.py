import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load file
df = pd.read_excel("/mnt/d/DATASETS/hysterysisCycle.xlsx")

# columns
x_col = "Strain %"
y_col = "Stress, Mpa"
cycle_col = "Cycle"

areas = {}

for cycle_id, g in df.groupby(cycle_col):

    x = g[x_col].to_numpy()
    y = g[y_col].to_numpy()

    # close loop (MANDATORY)
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # shoelace formula
    area = 0.5 * abs(
        np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])
    )

    areas[cycle_id] = area

    # optional plot per cycle
    plt.plot(x, y, label=f"Cycle {cycle_id}")

plt.xlabel("Stress (MPa)")
plt.ylabel("Strain (%)")
plt.title("Stress–Strain Hysteresis Loops")
plt.grid(True)
plt.legend()
plt.show()

# print results
for c, a in areas.items():
    print(f"Cycle {c}: Area = {a}")
