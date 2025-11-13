# eps_sweep_and_examples.py
import numpy as np
import os
from pathlib import Path
from data_loaders import get_dataloaders
from attack_and_evaluate import classifier, model  # requires attack_and_evaluate to expose classifier/model
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from PIL import Image
import csv

# Create output directory
os.makedirs("sweep_out", exist_ok=True)

# Load test dataset
_, _, test_loader = get_dataloaders(batch_size=16)
xs, ys = [], []
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("Loading test dataset for eps sweep...")
for imgs, labels in test_loader:
    # Denormalize for ART
    imgs_np = imgs.numpy() * std.reshape((1, 3, 1, 1)) + mean.reshape((1, 3, 1, 1))
    xs.append(imgs_np)
    ys.append(labels.numpy())

x_all = np.concatenate(xs, 0)
y_all = np.concatenate(ys, 0)
print(f"Loaded {x_all.shape[0]} test samples")

# Extended epsilon sweep for FGSM
eps_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.075, 0.1]
fgsm_results = []

print("\nRunning extended FGSM epsilon sweep...")
for eps in eps_list:
    print(f"  FGSM eps={eps}...")
    atk = FastGradientMethod(estimator=classifier, eps=eps)
    x_adv = atk.generate(x=x_all)
    preds = classifier.predict(x_adv).argmax(axis=1)
    acc = (preds == y_all).mean()
    fgsm_results.append((eps, acc))
    print(f"    Accuracy: {acc:.4f}")
    
    # Save first 5 adversarial examples for this epsilon
    for i in range(min(5, x_all.shape[0])):
        # Clip and convert to uint8
        im = np.clip(x_adv[i], 0, 1)
        im = (im * 255).astype('uint8')
        # Convert from CHW to HWC format
        im_hwc = np.transpose(im, (1, 2, 0))
        # Convert to PIL and save
        Image.fromarray(im_hwc).save(f"sweep_out/fgsm_eps{eps:.3f}_{i}.png")

# PGD with different iteration counts
print("\nRunning PGD with different iteration counts...")
pgd_results = []
pgd_iterations = [10, 20, 40, 80]

for max_iter in pgd_iterations:
    print(f"  PGD eps=0.03, max_iter={max_iter}...")
    pgd = ProjectedGradientDescent(
        estimator=classifier, 
        eps=0.03, 
        eps_step=0.005,  # eps/6 for better convergence
        max_iter=max_iter,
        random_eps=True
    )
    x_adv_pgd = pgd.generate(x=x_all)
    preds_pgd = classifier.predict(x_adv_pgd).argmax(axis=1)
    acc_pgd = (preds_pgd == y_all).mean()
    pgd_results.append((max_iter, acc_pgd))
    print(f"    Accuracy: {acc_pgd:.4f}")
    
    # Save examples for the strongest PGD (max iterations)
    if max_iter == max(pgd_iterations):
        for i in range(min(10, x_all.shape[0])):
            im = np.clip(x_adv_pgd[i], 0, 1)
            im = (im * 255).astype('uint8')
            im_hwc = np.transpose(im, (1, 2, 0))
            Image.fromarray(im_hwc).save(f"sweep_out/pgd_eps0.03_iter{max_iter}_{i}.png")

# Save original images for comparison
print("\nSaving original images for comparison...")
for i in range(min(10, x_all.shape[0])):
    im = np.clip(x_all[i], 0, 1)
    im = (im * 255).astype('uint8')
    im_hwc = np.transpose(im, (1, 2, 0))
    Image.fromarray(im_hwc).save(f"sweep_out/original_{i}.png")

# Save results to CSV files
print("\nSaving results...")

# FGSM sweep results
with open("fgsm_eps_sweep.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epsilon", "accuracy"])
    writer.writerows(fgsm_results)

# PGD iteration results
with open("pgd_iteration_sweep.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["max_iterations", "accuracy"])
    writer.writerows(pgd_results)

# Combined summary
with open("attack_sweep_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["attack_type", "parameter", "parameter_value", "accuracy"])
    
    for eps, acc in fgsm_results:
        writer.writerow(["FGSM", "epsilon", eps, acc])
    
    for max_iter, acc in pgd_results:
        writer.writerow(["PGD", "max_iterations", max_iter, acc])

print("Results saved to:")
print("  - fgsm_eps_sweep.csv")
print("  - pgd_iteration_sweep.csv") 
print("  - attack_sweep_summary.csv")
print("  - Adversarial examples in sweep_out/ directory")

# Print summary statistics
print(f"\nSummary:")
print(f"FGSM: eps range {min(eps_list):.3f}-{max(eps_list):.3f}, accuracy range {min(acc for _, acc in fgsm_results):.3f}-{max(acc for _, acc in fgsm_results):.3f}")
print(f"PGD: iter range {min(pgd_iterations)}-{max(pgd_iterations)}, accuracy range {min(acc for _, acc in pgd_results):.3f}-{max(acc for _, acc in pgd_results):.3f}")
print(f"Saved {len(eps_list) * 5 + 10 + 10} adversarial example images in sweep_out/")