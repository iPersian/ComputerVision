import torch
from pathlib import Path
import numpy as np
from utils import seed_all, EarlyStopping, get_model, save_model, load_model
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from dataset import get_dataloader
import argparse


def validate_batch(loader, model, device):
    model = model.to(device)
    model.eval()
    gts = []
    preds = []
    with torch.no_grad():
        with tqdm(loader, total=len(loader)) as tepoch:
            for inputs, target in tepoch:
                inputs = inputs.float().to(device)
                targets = target.long().to(device)
                outputs = model(inputs)
                # print("input")
                predictions = torch.argmax(outputs, dim=1)
                # print(targets)
                # print(predictions)
                preds.extend(predictions.cpu().numpy())
                gts.extend(targets.cpu().numpy())
    f1_s = f1_score(np.array(gts), np.array(preds), average="weighted")
    print(f1_s)
    return f1_s


def train_epoch(train_loader, model, optimizer, criterion, epoch):
    running_loss = 0.0
    train_l = tqdm(train_loader, total=len(train_loader))
    for imgs, lbls in train_l:
        imgs = imgs.to("cuda")
        lbls = lbls.to("cuda")

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        print(f'Loss: {loss}')
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_l.set_description_str(f"[{epoch + 1}] loss: {running_loss:.3f}")

    return


if __name__ == "__main__":
    seed_all(42)
    train_folder = Path("patches/train")
    val_folder = Path("patches/validation")
    class_map_f = Path("../data/class_map.txt")
    model_save_path = Path("./models")
    model_save_path.mkdir(parents=True, exist_ok=True)
    log_path = Path("./logs")
    log_path.mkdir(parents=True, exist_ok=True)
    img_size = (224, 224)
    n_workers = 4
    fname = f"1_efficientnet_b0"
    val_log = []

    train_loaders, val_loaders, labels = get_dataloader(
        train_folder, val_folder, class_map_f, batch_size=100, img_size=img_size, n_workers=n_workers
    )

    # model = get_model("efficientnet_b0", pretrained=True, num_classes=len(labels))
    model = load_model(Path("models"), Path("1_efficientnet_b0"))

    early_stop = EarlyStopping(patience=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", verbose=True, factor=0.5, min_lr=1e-5, patience=5
    )
    model = model.to("cuda")
    model.train()

    # f1_s_ = validate_batch(train_loaders, model, device="cuda")
    # print(f1_s_)

    for epoch in range(20):
        train_epoch(train_loaders, model, optimizer, criterion, epoch)

        if epoch % 1 == 0:
            f1_s = validate_batch(val_loaders, model, device="cuda")
            # val_log.append([epoch, f1_s])
            # early_stop(f1_s)
            # scheduler.step(metrics=f1_s)
            # if early_stop.do_stop:
            #     print(f"stopped early at epoch: {epoch}")
            #     break
    save_model(model, "efficientnet_b0", len(labels), 1, 1e-3, 0.5, save_path=model_save_path, fname=fname)
    # val_log = np.asarray(val_log)
    # print(val_log)
