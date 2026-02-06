import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("/mnt/d/DATASETS/hysterysisCycle.xlsx")

y_col = "Stress, Mpa"
x_col = "Strain %"
cycle_col = "Cycle"

results = []

plt.figure(figsize=(6, 5))

for cycle_id, g in df.groupby(cycle_col):

    x = g[x_col].to_numpy()
    y = g[y_col].to_numpy()

    # close loop
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # area
    area = 0.5 * abs(
        np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])
    )

    results.append({
        "Cycle": cycle_id,
        "Area": area
    })

    # plot loops (transparent)
    plt.plot(x, y, color="black", alpha=0.05)

plt.ylabel("Stress (MPa)")
plt.xlabel("Strain (%)")
plt.title("Stress–Strain Hysteresis Loops (All Cycles)")
plt.grid(True)
plt.show()

area_df = pd.DataFrame(results)
area_df.to_excel("cycle_area.xlsx", index=False)

plt.figure(figsize=(7, 4))
plt.plot(area_df["Cycle"], area_df["Area"], "-o", markersize=3)
plt.xlabel("Cycle")
plt.ylabel("Hysteresis Area")
plt.title("Energy Dissipation per Cycle")
plt.grid(True)
plt.show()
