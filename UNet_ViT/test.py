from dataset import version
import torch
from PIL import Image
from torchvision import transforms
import random
import model
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import filedialog
import pandas as pd
import numpy as np

device = 'cuda'

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

val_ds = pd.read_csv(f"../dataset{version}/val{version}.csv")

def _load_image(path):
    return to_tensor(Image.open(Path("../") / path))

inputs_p, label_p = val_ds.iloc[random.randint(0, len(val_ds) - 1)]
inputs_p = inputs_p.split(",")
label = _load_image(Path(label_p))
inputs = [_load_image(Path(img)) for img in inputs_p]
inputs = torch.stack(inputs, dim=0).squeeze(1).to(device)

inputs_rgb_p = [Path(i.replace("masks_small", "jpegs_small")).with_suffix(".jpeg") for i in inputs_p]
inputs_rgb = [_load_image(img) for img in inputs_rgb_p]
label_rgb = _load_image(Path(label_p.replace("masks_small", "jpegs_small")).with_suffix(".jpeg"))

# test_img = to_tensor(Image.open(Path(img_p))).to(device)
# test_mask = to_tensor(Image.open(mask_p)).to(device)

model = model.Network().to(device)
model.eval()
checkpoint = torch.load(f'epoch_34_model_only.pth')
model.load_state_dict(checkpoint['model_state'])

prediction = torch.round(model(inputs.unsqueeze(0)).squeeze(0))

diff = prediction - inputs[3]

# Set up the figure for matplotlib with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

# Show original images
fourth_mask = inputs[3]
fourth_rgb = inputs_rgb[3]

axes[0, 0].imshow(to_pil(fourth_mask), cmap='gray')
axes[0, 0].set_title('Input Mask 4')
axes[0, 0].axis('off')

# axes[0, 1].imshow(to_pil(test_img[1]), cmap='gray')
# axes[0, 1].set_title('Original Image 2')
# axes[0, 1].axis('off')
#
# axes[0, 2].imshow(to_pil(test_img[2]), cmap='gray')
# axes[0, 2].set_title('Original Image 3')
# axes[0, 2].axis('off')
#
# axes[1, 0].imshow(to_pil(test_img[3]), cmap='gray')
# axes[1, 0].set_title('Original Image 4')
# axes[1, 0].axis('off')

axes[0, 1].imshow(to_pil(prediction), cmap='gray')
axes[0, 1].set_title('Predicted Mask')
axes[0, 1].axis('off')

axes[0, 2].imshow(to_pil(diff), cmap='gray')
axes[0, 2].set_title('Growth Mask')
axes[0, 2].axis('off')

# # Show prediction image
# axes[1, 1].imshow(to_pil(prediction), cmap='gray')
# axes[1, 1].set_title('Predicted Image')
# axes[1, 1].axis('off')
#
# # Show difference image
# axes[1, 2].imshow(to_pil(diff), cmap='gray')
# axes[1, 2].set_title('Difference Image')
# axes[1, 2].axis('off')

axes[1,0].imshow(to_pil(fourth_rgb))
axes[1,0].set_title('RGB Image 4')
axes[1,0].axis('off')

alpha = 0.5  # Transparency level
for i in range(3):  # Dim the image
    fourth_rgb[i] = (1 - alpha) * fourth_rgb[i]

# Apply a red mask
fourth_rgb[0] += (alpha * diff.cpu()).squeeze(0)

axes[1,1].imshow(to_pil(fourth_rgb.cpu()))
axes[1,1].set_title('Growth visualized Image')
axes[1,1].axis('off')

axes[1,2].imshow(to_pil(label_rgb))
axes[1,2].set_title('Ground truth RGB Image')
axes[1,2].axis('off')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()