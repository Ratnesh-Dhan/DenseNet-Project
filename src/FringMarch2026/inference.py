import numpy as np
import torch
from services.model import UNet
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

model = UNet()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
model.cuda()

for i in tqdm(range(10,31)):
    img = cv2.imread(f"/mnt/d/DATASETS/mntFiles/bmp/{i}.bmp",0)
    # img = img/255.0
    img = img.astype(np.float32) / 255.0

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().cuda()

    # pred = model(tensor)
    with torch.no_grad():
        pred = model(tensor)
    # height = pred.squeeze().cpu().detach().numpy()
    height = pred.squeeze().cpu().numpy()
    # plt.plot(height[512][:])
    # plt.show()

    plt.imshow(height, cmap='jet')
    plt.axis("off")
    plt.title(f"{i}.bmp")
    plt.savefig(f"./saved/{i}.png")
    # exit(0)
    # np.save("pred_height.npy",height)
    # or
    np.savetxt(f"./saved/{i}_pred_height.csv",height,delimiter=",")