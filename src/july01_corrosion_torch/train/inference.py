import matplotlib.pyplot as plt
import random

model.eval()
img, _ = dataset_test[0]
with torch.no_grad():
    prediction = model([img.to(device)])

# Visualize masks
img = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()
plt.imshow(img)
for mask in prediction[0]['masks']:
    mask = mask[0].mul(255).byte().cpu().numpy()
    plt.imshow(mask, alpha=0.5)
plt.axis('off')
plt.show()
