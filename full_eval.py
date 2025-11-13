# full_eval.py
import numpy as np, torch
from pathlib import Path
from data_loaders import get_dataloaders
from attack_and_evaluate import classifier  # reuse the already-configured classifier object
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
import csv

# Load entire test dataset
_, _, test_loader = get_dataloaders(batch_size=16)
xs, ys = [], []
mean = np.array([0.485,0.456,0.406], dtype=np.float32)
std  = np.array([0.229,0.224,0.225], dtype=np.float32)

print("Loading test dataset...")
for imgs, labels in test_loader:
    # Denormalize images for ART attacks
    imgs_np = imgs.numpy() * std.reshape((1,3,1,1)) + mean.reshape((1,3,1,1))
    xs.append(imgs_np.astype(np.float32))
    ys.append(labels.numpy())

x_all = np.concatenate(xs, axis=0)
y_all = np.concatenate(ys, axis=0)
print(f"Total test samples: {x_all.shape[0]}")

# Evaluate clean accuracy first
print("\nEvaluating clean accuracy...")
clean_preds = classifier.predict(x_all).argmax(axis=1)
clean_acc = (clean_preds == y_all).mean()
print(f"Clean accuracy: {clean_acc:.4f}")

# FGSM sweep across different epsilon values
eps_list = [0.01, 0.02, 0.03, 0.05]
rows = [("Clean", 0.0, clean_acc)]

print("\nRunning FGSM attacks...")
for eps in eps_list:
    print(f"  FGSM eps={eps}...")
    atk = FastGradientMethod(estimator=classifier, eps=eps)
    x_adv = atk.generate(x=x_all)
    preds = classifier.predict(x_adv).argmax(axis=1)
    acc = (preds == y_all).mean()
    rows.append(("FGSM", eps, acc))
    print(f"    FGSM eps={eps} accuracy: {acc:.4f}")

# PGD attack with stronger settings
print("\nRunning PGD attack...")
pgd = ProjectedGradientDescent(
    estimator=classifier, 
    eps=0.03, 
    eps_step=0.005, 
    max_iter=20,
    random_eps=True
)
x_adv_pgd = pgd.generate(x=x_all)
pgd_preds = classifier.predict(x_adv_pgd).argmax(axis=1)
acc_pgd = (pgd_preds == y_all).mean()
rows.append(("PGD", 0.03, acc_pgd))
print(f"PGD eps=0.03 accuracy: {acc_pgd:.4f}")

# Save results to CSV
print("\nSaving results...")
with open("attack_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["attack_type", "epsilon", "accuracy"])
    writer.writerows(rows)

print("Results saved to attack_results.csv")

# Additional analysis: per-class breakdown
print("\nPer-class analysis:")
print("Clean accuracy by class:")
for class_id in [0, 1]:  # Normal=0, Pneumonia=1
    class_mask = y_all == class_id
    class_acc = (clean_preds[class_mask] == y_all[class_mask]).mean()
    class_name = "Normal" if class_id == 0 else "Pneumonia"
    print(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} samples)")

print("\nPGD attack accuracy by class:")
for class_id in [0, 1]:
    class_mask = y_all == class_id
    class_acc = (pgd_preds[class_mask] == y_all[class_mask]).mean()
    class_name = "Normal" if class_id == 0 else "Pneumonia"
    print(f"  {class_name}: {class_acc:.4f}")

# Confusion matrix for PGD
from sklearn.metrics import confusion_matrix
print("\nConfusion matrix for PGD attack:")
cm = confusion_matrix(y_all, pgd_preds)
print("True\\Pred  Normal  Pneumonia")
print(f"Normal     {cm[0,0]:6d}  {cm[0,1]:9d}")
print(f"Pneumonia  {cm[1,0]:6d}  {cm[1,1]:9d}")

print("\nFull evaluation complete!")