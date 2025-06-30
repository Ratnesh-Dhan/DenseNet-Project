from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch

def get_transform():
    return T.Compose([T.ToTensor()])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load dataset
dataset = CustomDataset("data/train", transforms=get_transform())
dataset_test = CustomDataset("data/val", transforms=get_transform())

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load model
num_classes = 2  # Background + 1 object
model = get_model_instance_segmentation(num_classes)
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        i += 1
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}], Loss: {losses.item():.4f}")
