from pathlib import Path

from mpmath.functions.zetazeros import count_to
from torch.utils.data import Dataset, sampler, DataLoader
from PIL import Image
from torchvision import transforms
import albumentations as a
import numpy as np
from collections import Counter

class MFWDDataset(Dataset):
    def __init__(self, data_folder: Path, labels_idx: dict, img_size: tuple, transforms):
        self.data_folder = data_folder
        self.img_size = img_size
        self.transforms = transforms
        self.labels_idx: dict = labels_idx
        self.img_list: list[Path] = sorted(list(data_folder.glob("*/*")))
        self.img_list = [entry for entry in self.img_list if entry.parent.stem in self.labels_idx]
        self.labels = [lbl.parent.stem for lbl in self.img_list]

    def get_class_distribution(self):
        return Counter(self.labels)

    def calculate_label_weights(self):
        count_dict = self.get_class_distribution()
        return list(map(lambda x: 1 / count_dict[x], self.labels))

    def _load_images(self, img_path):
        img = Image.open(str(img_path))
        img = transforms.Resize(self.img_size)(img)
        return np.asarray(img)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = img_path.parent.stem
        label_id = self.labels_idx[label]
        img = self._load_images(img_path)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img, label_id



def load_class_map(class_map_f: Path):
    with open(class_map_f, "r") as f:
        class_dict = {lbl:idx for idx, lbl in enumerate(f.read().strip().split("\n"))}
    return class_dict

def get_dataloader(train_folder, val_folder, class_map_f, batch_size, img_size, n_workers):
    labels = load_class_map(class_map_f)
    means = (0.3300823, 0.32133222, 0.14781028)
    stds = (0.20910876, 0.24810849, 0.08084098)

    train_transform = a.Compose([
        a.HorizontalFlip(),
        a.VerticalFlip(),
        a.RandomRotate90(),
        a.Transpose(),
        a.Normalize(
            mean=means,
            std=stds,
            max_pixel_value=255
        ),
        a.ToTensorV2()
    ])

    val_transform = a.Compose([
        a.Normalize(
            mean=means,
            std=stds,
            max_pixel_value=255
        ),
        a.ToTensorV2()
    ])

    train_dataset = MFWDDataset(train_folder, labels, img_size, train_transform)
    validation_dataset = MFWDDataset(val_folder, labels, img_size, val_transform)
    # print(len(train_dataset))

    sample_weights = train_dataset.calculate_label_weights()
    weighted_random_sampler = sampler.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )
    t_loader = DataLoader(train_dataset, sampler=weighted_random_sampler, batch_size=batch_size, num_workers=n_workers)
    v_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return t_loader, v_loader, labels

# train, val = get_dataloader(Path("patches/train"), Path("patches/validation"), Path("./data/class_map.txt"), 500, (224, 224), 1)
# print(len(train))