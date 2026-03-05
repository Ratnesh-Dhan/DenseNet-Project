import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from services.model import UNet
from services.patcher import FringeDataset
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

dataset = FringeDataset("/home/zumbie/Codes/NML/Datasets/bmp","/home/zumbie/Codes/NML/Datasets/heightmaps")
train_losses = []
val_losses = []
train_rmse_list = []
val_rmse_list = []
train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# loader = DataLoader(dataset,batch_size=4,shuffle=True)

model = UNet().cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
loss_fn = torch.nn.L1Loss()

best_val_loss = float("inf")
patience = 10
counter = 0

for epoch in range(100):
    #-------------TRAIN-------------
    model.train()
    train_loss = 0
    train_rmse = 0
    for img, height in tqdm(train_loader):

        img = img.cuda()
        height = height.cuda()

        pred = model(img)

        loss = loss_fn(pred,height)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_rmse += torch.sqrt(torch.mean((pred-height)**2)).item()


    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    train_rmse_list.append(train_rmse)

    #----------VALIDATION--------------
    print("Validation starts ............................")
    model.eval()
    val_loss = 0
    val_rmse = 0
    with torch.no_grad():
        for img, height in val_loader:
            img = img.cuda()
            height = height.cuda()

            pred = model(img)
            loss = loss_fn(pred, height)
            val_loss += loss.item()
            val_rmse += torch.sqrt(torch.mean((pred-height)**2)).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_rmse_list.append(val_rmse)
    print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    # ---------------CHECKPOINT------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model")
    else:
        counter += 1

    # --------------------EARLY STOP------------------------
    if counter >= patience:
        print("Early stopping triggered")
        break

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("loss.png")
plt.show()

plt.plot(train_rmse_list,label="Train RMSE")
plt.plot(val_rmse_list,label="Val RMSE")
plt.legend()
plt.title("RMSE per Epoch")
plt.savefig("rmse.png")

plt.subplot(1,2,1)
plt.title("Ground Truth")
plt.imshow(height.cpu()[0,0],cmap="jet")

plt.subplot(1,2,2)
plt.title("Prediction")
plt.imshow(pred.cpu()[0,0],cmap="jet")
plt.savefig("Predictoin vs ground truth.png")

error = (pred-height).cpu()[0,0]
plt.imshow(error,cmap="bwr")
plt.colorbar()
plt.title("Prediction Error")
plt.savefig("Error map.png")

err = error.numpy().flatten()
plt.hist(err,bins=100)
plt.title("Error Distribution")
plt.savefig("Histogram_of_errors.png")

gt = height.cpu().numpy().flatten()
pd = pred.cpu().detach().numpy().flatten()
r2 = r2_score(gt,pd)
print("R2:",r2)

plt.scatter(gt,pd,s=1)
plt.xlabel("True Height")
plt.ylabel("Predicted Height")
plt.title("Prediction vs True")
plt.savefig("Scatter_plot.png")