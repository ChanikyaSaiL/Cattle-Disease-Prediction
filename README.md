# Cattle Disease Prediction

**A deep learning system for multi-label cattle disease classification using hybrid CNN architectures.**

![Status](https://img.shields.io/badge/status-active-green)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

---

## 📋 Overview

This project implements a multi-label disease classification system for cattle using a hybrid deep learning approach. The model combines:
- **EfficientNet B0** for global feature extraction (1280 dims)
- **Custom Patch CNN** for local feature extraction (128 dims)
- **Attention-based fusion** layer to intelligently combine both feature sets

The system can detect up to **7 different disease/condition categories**:
1. BRD (Bovine Respiratory Disease)
2. Bovine
3. Disease
4. Respiratory
5. Healthy
6. Lumpy
7. Skin

---

## 🎯 Key Features

✅ **Multi-label Classification** - Single image can have multiple disease labels  
✅ **Hybrid Architecture** - Combines global (EfficientNet) + local (Patch CNN) features  
✅ **Attention Fusion** - Learnable weighted combination of feature sets  
✅ **Stratified Splitting** - Maintains label distribution across train/val/test  
✅ **EfficientNet B0** - Pretrained backbone for transfer learning  
✅ **High-resolution Input** - 224×224 image processing  
✅ **Auto-tuned Thresholds** - Per-class decision threshold optimization  

---

## 📁 Project Structure

```
Cattle-Disease-Prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── train.py                          # Training script
├── test.py                           # Testing & evaluation script
│
├── configs/
│   └── config.py                     # Configuration (paths, hyperparameters)
│
├── data/
│   ├── dataset.py                    # Custom Dataset class
│   ├── prepare_balanced_splits.py    # Data splitting utility
│   └── balanced_splits/              # CSV split files
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
│
├── models/
│   ├── dual_model.py                 # DualModel (fusion architecture)
│   └── patch_cnn.py                  # PatchCNN model
│
├── utils/
│   ├── augmentations.py              # Image augmentations
│   ├── loss.py                       # Custom loss functions (Focal Loss)
│   ├── mixup.py                      # Mixup augmentation
│   ├── temperature_scaling.py        # Calibration
│   ├── threshold.py                  # Threshold finding
│   ├── gradcam.py                    # Visualization
│   └── tta.py                        # Test-Time Augmentation
│
└── results/                          # Output directory (auto-generated)
    ├── models/                       # Trained model weights
    ├── metrics/                      # Per-class evaluation metrics
    ├── thresholds/                   # Decision thresholds
    ├── predictions/                  # Test predictions
    ├── summaries/                    # Results summaries
    └── README.md                     # Results documentation
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Cattle-Disease-Prediction
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Configuration

Edit `configs/config.py` to adjust hyperparameters:
```python
NUM_CLASSES = 7          # Number of disease categories
IMAGE_SIZE = 224         # Input image resolution
BATCH_SIZE = 16          # Batch size for training
EPOCHS = 25              # Number of training epochs
LR = 1e-4               # Learning rate
```

### Dataset Setup

1. **Prepare your data:**
   - Place images in `cattle diseases.v2i.multiclass/` directory with subdirectories: `train/`, `valid/`, `test/`
   - Each subdirectory should have a `_classes.csv` with image names and multi-hot encoded labels

2. **Run data balancing (if needed):**
```bash
python data/prepare_balanced_splits.py
```

This creates balanced CSV files in `data/balanced_splits/`

---

## 🏋️ Training

Train the fusion model with automatic checkpoint saving and threshold tuning:

```bash
python train.py
```

**Output:**
- Trained model: `results/models/fusion_model.pth`
- Decision thresholds: `results/thresholds/fusion_thresholds.npy`

**Training Features:**
- Early stopping via best validation loss
- Automatic checkpoint saving
- Per-class threshold optimization on validation set
- Threshold clipping (0.2-0.8) to prevent model collapse
- MixUp augmentation for regularization

---

## 🧪 Testing & Evaluation

Evaluate the trained model on test set and compute metrics:

```bash
python test.py
```

**Output:**
- Per-class accuracy: `results/metrics/fusion_per_class_acc.npy`
- Per-class F1 scores: `results/metrics/fusion_per_class_f1.npy`
- Summary metrics (JSON): `results/summaries/fusion_results.json`
- Summary metrics (NumPy): `results/summaries/fusion_results.npy`

**Computed Metrics:**
- Subset Accuracy (all labels correct)
- F1 Micro/Macro (weighted and unweighted)
- Hamming Loss
- ROC-AUC (Macro)
- PR-AUC (Macro)
- Per-class accuracy and F1 scores

---

## 📊 Model Architecture

### Dual Model (Fusion)

```
Input Image (3, 224, 224)
    ↓
    ├─→ EfficientNet B0 ──→ BatchNorm(1280) ──→ f1
    │                                           ↓
    └─→ Patch CNN ────────→ BatchNorm(128) ──→ f2
                                                ↓
                            Projection(128→1280)
                                                ↓
                            Attention Fusion ──→ fused(1280)
                                                ↓
                            Dense Layer ────────→ logits(7)
                                                ↓
                            Sigmoid ────────────→ probabilities(7)
```

### Key Components

**EfficientNet B0:**
- Pretrained on ImageNet
- Global feature extraction (1280 channels)
- Transfer learning backbone

**Patch CNN:**
- Custom lightweight architecture
- Local fine-grained features
- 3 convolutional blocks with pooling

**Attention Fusion:**
- Projects both features to 1280D
- Learns weighted combination: `output = w1*f1 + w2*f2`
- Softmax weights from feature concatenation

---

## 📈 Performance

The hybrid architecture provides benefits:
- **EfficientNet**: Captures high-level disease patterns
- **Patch CNN**: Captures fine-grained local details
- **Fusion**: Intelligently combines both perspectives

Typical results on balanced test set:
- Subset Accuracy: ~0.65-0.75
- F1 Macro: ~0.60-0.70
- Per-class F1 varies by disease prevalence

---

## 💾 Results Directory

All outputs are automatically organized:

| Directory | Contents | Examples |
|-----------|----------|----------|
| `results/models/` | Trained weights (.pth) | fusion_model.pth |
| `results/metrics/` | Per-class scores (.npy) | fusion_per_class_f1.npy |
| `results/thresholds/` | Decision thresholds (.npy) | fusion_thresholds.npy |
| `results/predictions/` | Test predictions (.npy) | predictions.npy, targets.npy |
| `results/summaries/` | Aggregated results | fusion_results.json |

See [results/README.md](results/README.md) for detailed format documentation.

---

## 📦 Dependencies

Core requirements:
- **torch** - Deep learning framework
- **torchvision** - Vision models and transforms
- **timm** - Additional pretrained models
- **scikit-learn** - Metrics and utilities
- **pandas** - Data handling
- **numpy** - Numerical computing
- **iterative-stratification** - Multi-label stratified splitting
- **openpyxl** - Excel support

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration Details

### Training Hyperparameters ([configs/config.py](configs/config.py))
```python
DEVICE = "cuda" if available else "cpu"
NUM_CLASSES = 7
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4  # Adam optimizer
```

### Data Paths
```python
BASE_DIR = "cattle diseases.v2i.multiclass"
TRAIN_EXCEL = "data/balanced_splits/train.csv"
VAL_EXCEL = "data/balanced_splits/valid.csv"  
TEST_EXCEL = "data/balanced_splits/test.csv"
```

### Output Paths
```python
FUSION_MODEL_PATH = "results/models/fusion_model.pth"
FUSION_THRESHOLD_PATH = "results/thresholds/fusion_thresholds.npy"
```

---

## 📚 Usage Examples

### Loading a Trained Model
```python
import torch
from models.dual_model import DualModel
from configs.config import *

model = DualModel(NUM_CLASSES, mode="fusion").to(DEVICE)
model.load_state_dict(torch.load("results/models/fusion_model.pth"))
model.eval()
```

### Making Predictions
```python
import torch
import numpy as np
from PIL import Image
from utils.augmentations import get_val_transforms

# Load image
image = Image.open("cow_image.jpg").convert("RGB")
transform = get_val_transforms()
x = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)
    
# Apply thresholds
thresholds = np.load("results/thresholds/fusion_thresholds.npy")
predictions = (probs.cpu().numpy() > thresholds).astype(int)
```

### Evaluating Results
```python
import json
import numpy as np

# Load metrics
with open("results/summaries/fusion_results.json") as f:
    metrics = json.load(f)

print(f"F1 Macro: {metrics['f1_macro']:.4f}")
print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")

# Per-class performance
acc = np.load("results/metrics/fusion_per_class_acc.npy")
f1 = np.load("results/metrics/fusion_per_class_f1.npy")
for i in range(7):
    print(f"Class {i}: Acc={acc[i]:.4f}, F1={f1[i]:.4f}")
```

---

## 🐛 Troubleshooting

**Issue:** `RuntimeError: The size of tensor a (1280) must match...`
- **Solution:** Check `NUM_CLASSES` in config.py matches the actual dataset labels

**Issue:** GPU out of memory
- **Solution:** Reduce BATCH_SIZE in config.py or use CPU (slower but no memory limit)

**Issue:** Low validation accuracy
- **Solution:** 
  - Check data quality and label accuracy
  - Adjust learning rate
  - Increase EPOCHS
  - Verify thresholds are being tuned properly

---

## 📖 Additional Resources

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) - Scaling CNNs Efficiently
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Deep Learning Framework
- [Multi-label Classification](https://scikit-learn.org/stable/modules/multiclass.html) - Sklearn Guide

---

## 📝 License

[MIT License](LICENSE) - Feel free to use this project for research and development.

---

## 👤 Author

Developed for cattle disease classification using modern deep learning techniques.

---

## 🙌 Acknowledgments

- **EfficientNet** - Pretrained model from TorchVision
- **Dataset** - Roboflow cattle diseases dataset
- **PyTorch** - Deep learning framework
- **Scikit-learn** - Evaluation metrics

---

## 📬 Contact & Support

For issues, suggestions, or collaborations:
- Review the [results/README.md](results/README.md) for output file formats
- Check [configs/config.py](configs/config.py) for all configuration options
- Inspect individual utility modules in `utils/` for advanced features

---

**Last Updated:** March 26, 2026  
**Version:** 1.0.0