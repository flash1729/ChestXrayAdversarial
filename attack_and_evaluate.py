# attack_and_evaluate.py
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from data_loaders import get_dataloaders
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

_, _, test_loader = get_dataloaders(batch_size=16)

# ---------------------------
# Simple model initialization
# ---------------------------
# Your fine-tuned chest X-ray model checkpoint (created by train.py)
finetuned_weights = Path("resnet18_chestxray.pth")

# Initialize ResNet18 without ImageNet weights
model = models.resnet18(weights=None)
# Replace final layer for our 2-class task
model.fc = nn.Linear(model.fc.in_features, 2)

# Load your fine-tuned checkpoint
if not finetuned_weights.exists():
    raise FileNotFoundError(f"Saved model {finetuned_weights} not found. Run train.py first.")
model.load_state_dict(torch.load(finetuned_weights, map_location="cpu"))
print("Loaded fine-tuned weights from:", finetuned_weights)

model = model.to(device)
model.eval()
# ---------------------------
# End model init
# ---------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

mean = np.array([0.485,0.456,0.406], dtype=np.float32)
std  = np.array([0.229,0.224,0.225], dtype=np.float32)

classifier = PyTorchClassifier(
    model=model, loss=criterion, optimizer=optimizer,
    input_shape=(3,224,224), nb_classes=2, clip_values=(0.0,1.0), preprocessing=(mean, std)
)

def collect_test(n_batches=10):
    xs, ys = [], []
    for i, (imgs, labels) in enumerate(test_loader):
        if i >= n_batches: break
        imgs_np = imgs.numpy()
        # dataloader returned normalized tensors; denormalize back to [0,1]
        m = mean.reshape((1,3,1,1)); s = std.reshape((1,3,1,1))
        imgs_np = imgs_np * s + m
        xs.append(imgs_np.astype(np.float32))
        ys.append(labels.numpy())
    if len(xs)==0: return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

x_test, y_test = collect_test(n_batches=20)
print("Collected test samples:", x_test.shape)

preds = classifier.predict(x_test)
acc_clean = (preds.argmax(axis=1) == y_test).mean()
print(f"Clean accuracy (sample): {acc_clean:.4f}")

# FGSM
fgsm = FastGradientMethod(estimator=classifier, eps=0.03)
x_adv_fgsm = fgsm.generate(x=x_test)
preds_fgsm = classifier.predict(x_adv_fgsm)
acc_fgsm = (preds_fgsm.argmax(axis=1) == y_test).mean()
print(f"FGSM acc: {acc_fgsm:.4f}")

# PGD
pgd = ProjectedGradientDescent(estimator=classifier, eps=0.03, eps_step=0.005, max_iter=20)
x_adv_pgd = pgd.generate(x=x_test)
preds_pgd = classifier.predict(x_adv_pgd)
acc_pgd = (preds_pgd.argmax(axis=1) == y_test).mean()
print(f"PGD acc: {acc_pgd:.4f}")

def norms(x_clean, x_adv):
    delta = x_adv - x_clean
    linf = np.max(np.abs(delta).reshape((delta.shape[0], -1)), axis=1).mean()
    l2 = np.linalg.norm(delta.reshape((delta.shape[0], -1)), axis=1).mean()
    return linf, l2

linf_fgsm, l2_fgsm = norms(x_test, x_adv_fgsm)
linf_pgd, l2_pgd = norms(x_test, x_adv_pgd)
print(f"FGSM mean L-inf: {linf_fgsm:.4f}, mean L2: {l2_fgsm:.4f}")
print(f"PGD  mean L-inf: {linf_pgd:.4f}, mean L2: {l2_pgd:.4f}")

os.makedirs("adv_out", exist_ok=True)
def save_examples(x_arr, prefix):
    for i in range(min(10, x_arr.shape[0])):
        x = np.clip(x_arr[i], 0.0, 1.0)
        x = (x * 255).astype(np.uint8)
        img = Image.fromarray(np.transpose(x, (1,2,0)))
        img.save(f"adv_out/{prefix}_adv_{i}.png")

save_examples(x_adv_fgsm, "fgsm")
save_examples(x_adv_pgd, "pgd")
print("Saved adversarial examples to adv_out/")
