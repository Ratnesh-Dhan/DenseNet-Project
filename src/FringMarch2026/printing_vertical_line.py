import numpy as np
import torch
from services.model import UNet
import cv2
from matplotlib import pyplot as plt
filepath = "/mnt/d/DATASETS/mntFiles/heightmaps/2_height_map.npy"
img = cv2.imread("/mnt/d/DATASETS/mntFiles/bmp/1.bmp",0)
# img = img/255.0
img = img.astype(np.float32) / 255.0

tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().cuda()

model = UNet()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
model.cuda()
# pred = model(tensor)
with torch.no_grad():
    pred = model(tensor)
# height = pred.squeeze().cpu().detach().numpy()
height = pred.squeeze().cpu().numpy()

heightmapO = np.load(filepath)
mean = heightmapO.mean()
std = heightmapO.std()

heightmapO = (heightmapO - mean) / std

vmin = min(heightmapO.min(), height.min())
vmax = max(heightmapO.max(), height.max())

print(heightmapO[512][:])
plt.subplot(2,2,1)
plt.imshow(heightmapO, cmap='jet', vmin=vmin, vmax=vmax)
plt.subplot(2,2,3)
plt.plot(heightmapO[512][:])

plt.subplot(2,2,2)
plt.imshow(height, cmap='jet', vmin=vmin, vmax=vmax)
plt.subplot(2,2,4)
plt.plot(height[512][:])
plt.savefig("4image.png")
plt.show()