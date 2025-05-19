import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mask = np.array(Image.open('path_to_mask.png'))
unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
print("Unique RGB colors in mask:", unique_colors)
