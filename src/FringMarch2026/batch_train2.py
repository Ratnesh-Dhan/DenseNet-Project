import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from services.model import UNet
from services.patcher import FringeDataset
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

# dataset = FringeDataset("/home/zumbie/Codes/NML/Datasets/bmp","/home/zumbie/Codes/NML/Datasets/heightmaps")
dataset = FringeDataset("/mnt/z/DATASETS/Fringe/bmp","/mnt/z/DATASETS/Fringe/heightmaps", patch=512)

train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

result_folder = "./batch_results_nonAMP_batch=8_LossFuntion_v2__lrScheduler"
os.makedirs(result_folder, exist_ok=True)

for optimizer_name in ["adam", "adamw", "sgd", "rmsprop"]:
    train_losses = []
    val_losses = []
    train_rmse_list = []
    val_rmse_list = []
    
    model = UNet().cuda()
    # model = torch.compile(model)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    # folder making
    result_dir = os.path.join(result_folder, optimizer_name)
    os.makedirs(result_dir, exist_ok=True)

    loss_fn = torch.nn.L1Loss()

    best_val_loss = float("inf")
    patience = 10
    counter = 0

    # AMP
    # scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    for epoch in range(100):
        #-------------TRAIN-------------
        model.train()
        train_loss = 0
        train_rmse = 0
        for img, height in tqdm(train_loader):

            img = img.cuda(non_blocking=True)
            height = height.cuda(non_blocking=True)

            optimizer.zero_grad()
            # NON AMP
            pred = model(img)
            # old style
            # loss = loss_fn(pred, height)
            l1 = torch.nn.functional.l1_loss(pred, height)
            pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            gt_dx = height[:, :, :, 1:] - height[:, :, :, :-1]

            pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            gt_dy = height[:, :, 1:, :] - height[:, :, :-1, :]

            grad_loss = torch.mean(torch.abs(pred_dx - gt_dx)) + \
                        torch.mean(torch.abs(pred_dy - gt_dy))

            loss = l1 + 0.1 * grad_loss
            loss.backward()
            optimizer.step()

            #AMP
            # with torch.amp.autocast('cuda'):
            #     pred = model(img)
            #     loss = loss_fn(pred,height)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            train_loss += loss.item()
            train_rmse += torch.sqrt(torch.mean((pred-height)**2))


        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_rmse_list.append(train_rmse.item())

        #----------VALIDATION--------------
        print("Validation starts ............................")
        model.eval()
        val_loss = 0
        val_rmse = 0
        with torch.no_grad():
            for img, height in val_loader:
                img = img.cuda(non_blocking=True)
                height = height.cuda(non_blocking=True)
                
                # NON AMP
                pred = model(img)
                # loss = loss_fn(pred, height)
                l1 = torch.nn.functional.l1_loss(pred, height)
                pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                gt_dx = height[:, :, :, 1:] - height[:, :, :, :-1]

                pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                gt_dy = height[:, :, 1:, :] - height[:, :, :-1, :]

                grad_loss = torch.mean(torch.abs(pred_dx - gt_dx)) + \
                            torch.mean(torch.abs(pred_dy - gt_dy))

                loss = l1 + 0.1 * grad_loss

                #AMP
                # with torch.amp.autocast('cuda'):
                #     pred = model(img)
                #     loss = loss_fn(pred, height)
                val_loss += loss.item()
                val_rmse += torch.sqrt(torch.mean((pred-height)**2))

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        val_rmse_list.append(val_rmse.item())
        print(f"Optimizer:{optimizer_name} Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        # ---------------CHECKPOINT------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))
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
    plt.savefig(os.path.join(result_dir, "loss.png"))
    plt.close()

    plt.plot(train_rmse_list,label="Train RMSE")
    plt.plot(val_rmse_list,label="Val RMSE")
    plt.legend()
    plt.title("RMSE per Epoch")
    plt.savefig(os.path.join(result_dir, "rmse.png"))
    plt.close()

    plt.subplot(1,2,1)
    plt.title("Ground Truth")
    plt.imshow(height.cpu()[0,0],cmap="jet")
    plt.subplot(1,2,2)
    plt.title("Prediction")
    plt.imshow(pred.cpu()[0,0],cmap="jet")
    plt.savefig(os.path.join(result_dir, "Predictoin vs ground truth.png"))
    plt.close()

    error = (pred-height).cpu()[0,0]
    plt.imshow(error,cmap="bwr")
    plt.colorbar()
    plt.title("Prediction Error")
    plt.savefig(os.path.join(result_dir, "Error map.png"))
    plt.close()

    err = error.numpy().flatten()
    plt.hist(err,bins=100)
    plt.title("Error Distribution")
    plt.savefig(os.path.join(result_dir, "Histogram_of_errors.png"))
    plt.close()

    gt = height.cpu().numpy().flatten()
    pd = pred.cpu().detach().numpy().flatten()
    r2 = r2_score(gt,pd)
    print("R2:",r2)
    with open (os.path.join(result_dir, "r_square.txt"), 'w') as f:
        f.write(f"R2: {r2}\n")
        f.write(f"Train RMSE: {train_rmse_list[-1]}\n")
        f.write(f"Val RMSE: {val_rmse_list[-1]}\n")
    f.close()

    plt.scatter(gt,pd,s=1)
    plt.xlabel("True Height")
    plt.ylabel("Predicted Height")
    plt.title("Prediction vs True")
    plt.savefig(os.path.join(result_dir, "Scatter_plot.png"))
    plt.close()