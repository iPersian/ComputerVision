import torch, gc
from criterion import IoULoss
from torch.optim import AdamW
from tqdm import tqdm
from model import *
from torch.utils.data import DataLoader
from dataset import get_training_ds, get_val_ds, version

device = 'cuda'

gc.collect()
torch.cuda.empty_cache()

def train(model, loader, criterion):
    model.eval()
    avg_loss = 0

    with torch.no_grad():
        loader = tqdm(enumerate(loader), total=len(loader))
        for idx, (data, target) in loader:
            data = data.to(device).float()
            target = target.to(device).float()

            output = model(data)

            loss = criterion(output, target)

            avg_loss += loss
    return avg_loss / len(loader)


print("Loading validation...")

criterion = IoULoss().to(device)
model = UNet().to(device)
model.eval()
checkpoint = torch.load(f'UNet_30{version}.pth')
model.load_state_dict(checkpoint['model_state'])

dataloader = DataLoader(get_val_ds(), batch_size=8, shuffle=True)

print("Starting...")

val_loss = train(model, dataloader, criterion)
print(f'Validation dataset loss: {val_loss}')