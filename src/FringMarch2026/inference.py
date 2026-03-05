import numpy as np
import torch
from services.model import UNet
import cv2

img = cv2.imread("test.bmp",0)
img = img/255.0

tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().cuda()

model = UNet().cuda()
pred = model(tensor)

height = pred.squeeze().cpu().detach().numpy()

np.save("pred_height.npy",height)
# or
np.savetxt("pred_height.csv",height,delimiter=",")