from dataset import val_ds
import torch
from torchvision import transforms
import random
import model
import matplotlib.pyplot as plt

device = 'cuda'

test_img = val_ds[random.randint(0, len(val_ds) - 1)][0].to(device)
to_pil = transforms.ToPILImage()

model = model.Network().to(device)
model.eval()
checkpoint = torch.load('epoch_29.pth')
model.load_state_dict(checkpoint['model_state'])

prediction = torch.round(model(test_img.unsqueeze(0)).squeeze(0))

diff = prediction - test_img[3]

# Set up the figure for matplotlib with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

# Show original images
axes[0, 0].imshow(to_pil(test_img[0]), cmap='gray')
axes[0, 0].set_title('Original Image 1')
axes[0, 0].axis('off')

axes[0, 1].imshow(to_pil(test_img[1]), cmap='gray')
axes[0, 1].set_title('Original Image 2')
axes[0, 1].axis('off')

axes[0, 2].imshow(to_pil(test_img[2]), cmap='gray')
axes[0, 2].set_title('Original Image 3')
axes[0, 2].axis('off')

axes[1, 0].imshow(to_pil(test_img[3]), cmap='gray')
axes[1, 0].set_title('Original Image 4')
axes[1, 0].axis('off')

# Show prediction image
axes[1, 1].imshow(to_pil(prediction), cmap='gray')
axes[1, 1].set_title('Predicted Image')
axes[1, 1].axis('off')

# Show difference image
axes[1, 2].imshow(to_pil(diff), cmap='gray')
axes[1, 2].set_title('Difference Image')
axes[1, 2].axis('off')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()