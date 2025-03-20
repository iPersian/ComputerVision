from pathlib import Path

import albumentations as a
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

data_folder = Path("patches/train")
bs = 5000

class MFWDDataset(Dataset):
    def __init__(self, data_folder: Path, labels: dict, img_size:tuple, transforms: transforms.Compose):
        self.data_folder = data_folder
        self.img_size = img_size
        self.transforms = transforms
        self.img_list = sorted(list(data_folder.glob("*/*")))
        self.labels = labels

    def _load_images(self, img_path):
        img = Image.open(str(img_path))
        img = transforms.Resize(self.img_size)(img)
        return img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = img_path.parent.stem
        label_id = self.labels[label]
        img = self._load_images(img_path)
        if self.transforms:
            img = self.transforms(img)
        return img.type(torch.float32), label_id

train_transform = transforms.Compose([
    transforms.PILToTensor(),
])

labels = {"ACHMI": 0,
          "AETCY": 1,
          "AGRRE": 2,
          "ALOMY": 3,
          "ARTVU": 4,
          "CHEAL": 5,
          "CIRAR": 6,
          "CONAR": 7,
          "ECHCG": 8,
          "GALAP": 9,
          "GASPA": 10,
          "GERMO": 11,
          "LAMAL": 12,
          "MATCH": 13,
          "PLAMA": 14,
          "POAAN": 15,
          "POLAM": 16,
          "POLCO": 17,
          "POROL": 18,
          "PULDY": 19,
          "SOLNI": 20,
          "SORVU": 21,
          "SSYOF": 22,
          "STEME": 23,
          "THLAR": 24,
          "VEROF": 25,
          "VIOAR": 26}

ds = MFWDDataset(data_folder=data_folder, labels= labels, img_size=(224, 224), transforms=train_transform)
loader = DataLoader(ds, batch_size=bs, shuffle=True)


mean = 0
std = 0
nb_samples = 0
_tqdm = tqdm(loader, total=len(loader))
for batch, labels in _tqdm:
    batch.to("cuda")
    B, C, H, W = batch.shape
    batch = batch.view(B, C, -1)
    mean += batch.mean(-1).sum(0)
    std += batch.std(-1).sum(0)
    nb_samples += B


mean /= nb_samples
std /= nb_samples

print(mean.numpy() / 255.0)
print(std.numpy() / 255.0)

