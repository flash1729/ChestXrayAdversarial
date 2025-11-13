# data_loaders.py
import os
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

def to_rgb(img):
    return img.convert("RGB")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Detect data directory robustly
BASE = Path("data")
CAND = BASE / "chest_xray"
if not CAND.exists():
    raise FileNotFoundError(f"Expected dataset at {CAND.resolve()}")

IMAGE_SIZE = 224
BATCH_SIZE = 16   # lower for Mac if memory is limited
NUM_WORKERS = 0

train_tf = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_tf = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def get_dataloaders(data_dir: str = str(CAND), batch_size: int = BATCH_SIZE):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    test_dir  = data_dir / "test"

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds   = ImageFolder(val_dir, transform=val_tf)
    test_ds  = ImageFolder(test_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print("Dataset classes:", train_ds.classes)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    tl, vl, ts = get_dataloaders()
    print("Train batches:", len(tl), "Val batches:", len(vl), "Test batches:", len(ts))

