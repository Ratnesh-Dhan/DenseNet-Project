import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from services.model import UNet
from services.patcher import FringeDataset

dataset = FringeDataset("images","heightmaps")

loader = DataLoader(dataset,batch_size=4,shuffle=True)

model = UNet().cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

loss_fn = torch.nn.L1Loss()

for epoch in range(50):

    loop = tqdm(loader)

    for img, height in loop:

        img = img.cuda()
        height = height.cuda()

        pred = model(img)

        loss = loss_fn(pred,height)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())