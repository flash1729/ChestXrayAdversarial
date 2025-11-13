# ChestXray Adversarial Robustness Analysis

A deep learning project that trains a ResNet18 model for chest X-ray classification (Normal vs Pneumonia) and evaluates its robustness against adversarial attacks using FGSM and PGD methods.

## 📋 Project Overview

This project implements a complete pipeline for:
1. **Medical Image Classification**: Binary classification of chest X-rays to detect pneumonia
2. **Adversarial Attack Generation**: Creating adversarial examples using state-of-the-art attack methods
3. **Robustness Evaluation**: Measuring model vulnerability to adversarial perturbations

## 🏗️ Project Structure

```
ChestXrayAdversarial/
├── README.md                    # Project documentation
├── data_loaders.py             # Data preprocessing and loading utilities
├── train.py                    # Model training and fine-tuning
├── attack_and_evaluate.py      # Adversarial attack generation and evaluation
├── eps_sweep_and_examples.py   # Epsilon parameter analysis
├── full_eval.py               # Comprehensive evaluation metrics
├── resnet18_chestxray.pth     # Trained model weights (generated after training)
├── attack_results.csv         # Attack evaluation results (generated)
├── data/
│   └── chest_xray/            # Dataset directory
│       ├── train/             # Training images
│       ├── val/               # Validation images
│       └── test/              # Test images
└── adv_out/                   # Generated adversarial examples (created after attacks)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- adversarial-robustness-toolbox (ART)
- PIL
- numpy
- tqdm
- pathlib

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/flash1729/ChestXrayAdversarial.git
cd ChestXrayAdversarial
```

2. **Install dependencies:**
```bash
pip install torch torchvision
pip install adversarial-robustness-toolbox
pip install Pillow numpy tqdm
```

3. **Download and prepare the dataset:**
   - Place the chest X-ray dataset in the `data/chest_xray/` directory
   - Ensure the directory structure follows the format:
     ```
     data/chest_xray/
     ├── train/
     │   ├── NORMAL/
     │   └── PNEUMONIA/
     ├── val/
     │   ├── NORMAL/
     │   └── PNEUMONIA/
     └── test/
         ├── NORMAL/
         └── PNEUMONIA/
     ```

### Usage

#### Step 1: Train the Model
```bash
python train.py
```
This will:
- Load and preprocess the chest X-ray dataset
- Train a ResNet18 model for 5 epochs using transfer learning
- Save the trained model as `resnet18_chestxray.pth`
- Display training progress and validation accuracy

#### Step 2: Generate Adversarial Attacks
```bash
python attack_and_evaluate.py
```
This will:
- Load the trained model
- Generate adversarial examples using FGSM and PGD attacks
- Evaluate model robustness on adversarial samples
- Save example adversarial images to `adv_out/` directory
- Display attack success rates and perturbation magnitudes

#### Step 3: Advanced Analysis (Optional)
```bash
python eps_sweep_and_examples.py    # Analyze different epsilon values
python full_eval.py                 # Comprehensive evaluation
```

## 🔧 Key Components

### Data Loader (`data_loaders.py`)
- **Image Preprocessing**: Converts to RGB, resizes to 224×224, applies normalization
- **Data Augmentation**: Random crops and horizontal flips for training
- **Batch Loading**: Efficient PyTorch DataLoader implementation
- **Reproducibility**: Fixed random seeds for consistent results

### Training Pipeline (`train.py`)
- **Transfer Learning**: Uses ResNet18 architecture with frozen backbone
- **Binary Classification**: Adapts final layer for Normal vs Pneumonia detection
- **Optimization**: Adam optimizer with step learning rate scheduling
- **Validation**: Per-epoch validation accuracy monitoring
- **Device Support**: Automatic MPS (Apple Silicon) and CPU support

### Adversarial Evaluation (`attack_and_evaluate.py`)
- **Attack Methods**:
  - **FGSM**: Fast single-step gradient-based attack
  - **PGD**: Iterative projected gradient descent attack
- **Robustness Metrics**: Clean vs adversarial accuracy comparison
- **Perturbation Analysis**: L∞ and L2 norm measurements
- **Visualization**: Saves adversarial examples for manual inspection

## 📊 Expected Results

### Model Performance
- **Clean Accuracy**: ~85-90% on test set (depending on dataset quality)
- **FGSM Attack (ε=0.03)**: ~30-50% accuracy drop
- **PGD Attack (ε=0.03)**: ~40-60% accuracy drop

### Adversarial Examples
- Generated adversarial images are saved in `adv_out/`
- Perturbations are typically imperceptible to human observers
- L∞ norm perturbations are bounded by the epsilon parameter (0.03)

## 🛠️ Configuration

### Key Parameters
- **Image Size**: 224×224 pixels (ImageNet standard)
- **Batch Size**: 16 (optimized for memory efficiency)
- **Training Epochs**: 5 (quick training for demonstration)
- **Attack Epsilon**: 0.03 (8/255 in pixel space)
- **PGD Iterations**: 20 steps

### Hardware Requirements
- **Memory**: 4GB+ RAM recommended
- **GPU**: Optional but recommended for faster training
- **Storage**: ~2GB for dataset + model weights

## 🔍 Advanced Features

### Epsilon Sweep Analysis
Run `eps_sweep_and_examples.py` to:
- Test multiple epsilon values (0.01, 0.02, 0.03, 0.05)
- Generate robustness curves
- Analyze attack effectiveness vs perturbation magnitude

### Comprehensive Evaluation
Run `full_eval.py` to:
- Evaluate on complete test set
- Generate detailed attack statistics
- Export results to CSV format

## 📈 Interpreting Results

### Robustness Indicators
- **High clean accuracy + low adversarial accuracy**: Model is not robust
- **Similar clean and adversarial accuracy**: Model shows some robustness
- **Large L∞/L2 norms**: Strong perturbations needed for successful attacks

### Medical Implications
- Adversarial vulnerabilities in medical imaging can have serious consequences
- This analysis helps understand model reliability under potential attacks
- Results inform the need for adversarial training or robust architectures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 References

- [Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ResNet Architecture](https://arxiv.org/abs/1512.03385)
- [FGSM Attack](https://arxiv.org/abs/1412.6572)
- [PGD Attack](https://arxiv.org/abs/1706.06083)

## 📞 Support

For questions or issues, please:
1. Check existing [GitHub Issues](https://github.com/flash1729/ChestXrayAdversarial/issues)
2. Open a new issue with detailed description
3. Include error messages and system information

---

**Note**: This project is for research and educational purposes. Medical applications require additional validation and regulatory compliance.