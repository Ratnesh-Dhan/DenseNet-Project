from tkinter import BASELINE
from turtle import title
import numpy as np
import torch
from services.model import UNet
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

base_path = "/mnt/g/DATASETS/FringeDataset"
model_path = "./batch_results"
optimizers = os.listdir(model_path)

images = os.listdir(os.path.join(base_path, "test_bmp"))
heigths = os.listdir(os.path.join(base_path, "test_height"))

for optimizer in tqdm(optimizers):
    saving_folder = os.path.join("./final_result", optimizer)
    os.makedirs(saving_folder, exist_ok=True)

    model = UNet()
    model.load_state_dict(torch.load(os.path.join(model_path, optimizer, "best_model.pth")))
    model.eval()
    model.cuda()

    for bmpImg in images:    
        i = bmpImg.split('.')[0]
        filepath = os.path.join(base_path, f"test_height/{i}_height_map.npy")
        img = cv2.imread(os.path.join(base_path, "test_bmp", bmpImg),0)
        # img = img/255.0
        img = img.astype(np.float32) / 255.0

        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().cuda()

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

        # print(heightmapO[512][:])
        
        plt.figure(figsize=(8, 10))
        plt.subplot(3,2,1)
        plt.imshow(img)
        plt.title("BMP image")
    
        plt.subplot(3,2,2)
        plt.imshow(img)
        plt.title("BMP image")

        plt.subplot(3,2,3)
        plt.imshow(heightmapO, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Height data of BMP image")
        plt.subplot(3,2,5)
        plt.plot(heightmapO[512][:])
        plt.ylim(bottom=0)

        plt.subplot(3,2,4)
        plt.imshow(height, cmap='jet', vmin=vmin, vmax=vmax)
        plt.title("Height data predicted")
        plt.subplot(3,2,6)
        plt.plot(height[512][:])
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(saving_folder, f"{i}_6_figure.png"))
        # plt.subplot(2,2,1)
        # plt.imshow(heightmapO, cmap='jet', vmin=vmin, vmax=vmax)
        # plt.title("Height data of input image")
        # plt.subplot(2,2,3)
        # plt.plot(heightmapO[512][:])
        # plt.ylim(bottom=0)

        # plt.subplot(2,2,2)
        # plt.imshow(height, cmap='jet', vmin=vmin, vmax=vmax)
        # plt.title("Height data output")
        # plt.subplot(2,2,4)
        # plt.plot(height[512][:])
        # plt.ylim(bottom=0)
        # plt.savefig(os.path.join(saving_folder, f"{i}_4_figure.png"))
        plt.close()