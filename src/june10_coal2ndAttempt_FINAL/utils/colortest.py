import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import sys, os


folderPath = fr"D:\NML 2nd working directory\DEEP SOUMYA 14-july-25\final\C3"
files = os.listdir(folderPath)

for file in files:
    image = plt.imread(os.path.join(folderPath, file))
    print(f"{file} : {image.shape}")







sys.exit(0)
# Your data
cavity_percentage = 10
cavity_filled_percentage = 20
inertinite_percentage = 25
minerals_percentage = 15
vitrinite_percentage = 30

# Create color patches
legend_patches = [
    mpatches.Patch(color='green', label=f'Cavity: {cavity_percentage}%'),
    mpatches.Patch(color='blue', label=f'Cavity Filled: {cavity_filled_percentage}%'),
    mpatches.Patch(color='red', label=f'Inertinite: {inertinite_percentage}%'),
    mpatches.Patch(color='yellow', label=f'Minerals: {minerals_percentage}%'),
    mpatches.Patch(color='purple', label=f'Vitrinite: {vitrinite_percentage}%')
]

# Create your plot (placeholder example)
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])  # Dummy plot
q
# Add legend at bottom center
plt.legend(
    handles=legend_patches,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.1),  # Adjust to your liking
    ncol=2,
    fontsize=12
)

plt.tight_layout()
plt.show()
