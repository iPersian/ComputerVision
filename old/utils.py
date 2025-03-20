import numpy as np
import random
import os
import torch
import timm


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.do_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score - self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.do_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def get_model(encoder_name, pretrained, num_classes):
    model = timm.create_model(encoder_name, pretrained=pretrained, num_classes=num_classes)
    return model

def save_model(model, encoder, num_classes, epoch, lr, lr_factor, save_path, fname):
    save_dict = {"model_sate_dict": model.state_dict(),
                 "encoder": encoder,
                 "num_classes": num_classes,
                 "epoch": epoch,
                 "lr": lr,
                 "lr_scheduler_factor": lr_factor}
    torch.save(save_dict, save_path/f"{fname}.pth")

def load_model(save_path, fname):
    loaded_model = torch.load(
        save_path/f"{fname}.pth", map_location=torch.device("cpu")
    )
    encoder = loaded_model["encoder"]
    num_classes = loaded_model["num_classes"]
    model = get_model(encoder, pretrained=False, num_classes=num_classes)
    model.load_state_dict(loaded_model["model_sate_dict"])
    return model

