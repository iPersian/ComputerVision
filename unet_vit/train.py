import torch, gc
from criterion import IoULoss
from torch.optim import AdamW
from tqdm import tqdm
from model import *
from torch.utils.data import Dataset, DataLoader
from dataset import get_training_ds
from torch.utils.tensorboard import SummaryWriter

device = 'cuda'

gc.collect()
torch.cuda.empty_cache()

# Create a writer to write to Tensorboard
writer = SummaryWriter()

def train(model, loader, criterion, optimizer, epoch):
    model.train()
    avg_loss = 0
    loader = tqdm(enumerate(loader), total=len(loader))
    for idx, (data, target) in loader:
        data = data.to(device).float()
        target = target.to(device).float()

        output = model(data)

        loss = criterion(output, target)
        writer.add_scalars(f"Loss epoch {epoch}", {'Train': loss}, idx)

        avg_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss / len(loader)


print("Loading...")

epochs = 5
criterion = IoULoss().to(device)
lr = 1e-3
model = Network().to(device)
optimizer = AdamW(model.parameters(), lr)

dataloader = DataLoader(get_training_ds(), batch_size=8, shuffle=True)

print("Starting...")
checkpoint = torch.load('epoch_29.pth')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])

for epoch in tqdm(range(30, 35)):
    train_loss = train(model, dataloader, criterion, optimizer, epoch)
    print(f'Train loss epoch {epoch}: {train_loss}')
    state = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict())
    torch.save(state, f'epoch_{epoch}.pth')

print('Finished Training')