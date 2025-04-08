from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as a
from pathlib import Path
import numpy as np
import torch
import pandas as pd

# version = "_1_1"
version = "_1_4"
# version = "_1_8"
# version = "_1_12"
class CustomDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms

    def _load_image(self, path):
        return np.asarray(Image.open(Path("../") / path))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        inputs, label = self.df.iloc[item]
        inputs = [Path(p) for p in inputs.split(",")]
        label = self._load_image(Path(label))
        inputs = [self._load_image(img) for img in inputs]
        if self.transforms:
            inputs = [self.transforms(image = img)["image"] / 255 for img in inputs]
            label = self.transforms(image = label)["image"] / 255

        return torch.stack(inputs, dim=0).squeeze(1), label


train_transform = a.Compose([
        a.HorizontalFlip(),
        a.VerticalFlip(),
        a.RandomRotate90(),
        a.Transpose(),
        # a.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        #     max_pixel_value=255
        # ),
        a.ToTensorV2()
    ])


val_transform = a.Compose([
#         a.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#             max_pixel_value=255
#         ),
        a.ToTensorV2()
    ])


training_ds = CustomDataset(pd.read_csv(f"../dataset{version}/train{version}.csv"), train_transform)
test_ds = CustomDataset(pd.read_csv(f"../dataset{version}/test{version}.csv"), train_transform)
val_ds = CustomDataset(pd.read_csv(f"../dataset{version}/val{version}.csv"), val_transform)


def get_training_ds():
    return training_ds

def get_val_ds():
    return val_ds