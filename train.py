# train.py
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from data_loaders import get_dataloaders

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

train_loader, val_loader, test_loader = get_dataloaders()

model = models.resnet18(weights=None)  # <-- no external download

# Freeze everything except the classifier head
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model.fc.requires_grad = True
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}")
    for imgs, labels in pbar:
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        pbar.set_postfix(loss=running_loss/running_total, acc=running_correct/running_total)

    scheduler.step()

    # validation
    model.eval()
    val_correct = 0; val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device); labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    print(f"Epoch {epoch+1} val acc: {val_correct/val_total:.4f}")

torch.save(model.state_dict(), "resnet18_chestxray.pth")
print("Saved model to resnet18_chestxray.pth")

