# Results Directory Structure

This directory contains all output files from model training, evaluation, and predictions.

## Folder Organization

```
results/
├── models/                         # Trained model weights
│   ├── best_model.pth             # EfficientNet model checkpoint
│   ├── eff_model.pth              # EfficientNet model weights
│   ├── fusion_model.pth            # Fusion model weights
│   └── patch_model.pth             # Patch CNN model weights
│
├── metrics/                        # Evaluation metrics
│   ├── eff_per_class_acc.npy       # Per-class accuracy (EfficientNet)
│   ├── eff_per_class_f1.npy        # Per-class F1 (EfficientNet)
│   ├── fusion_per_class_acc.npy    # Per-class accuracy (Fusion)
│   ├── fusion_per_class_f1.npy     # Per-class F1 (Fusion)
│   ├── patch_per_class_acc.npy     # Per-class accuracy (Patch)
│   ├── patch_per_class_f1.npy      # Per-class F1 (Patch)
│   ├── per_class_acc.npy           # Default per-class accuracy
│   └── per_class_f1.npy            # Default per-class F1
│
├── thresholds/                     # Decision thresholds
│   ├── eff_thresholds.npy          # EfficientNet thresholds
│   ├── fusion_thresholds.npy       # Fusion model thresholds
│   └── patch_thresholds.npy        # Patch CNN thresholds
│
├── predictions/                    # Prediction outputs
│   ├── eff_predictions.npy         # EfficientNet predictions
│   ├── eff_targets.npy             # Ground truth (EfficientNet eval)
│   ├── patch_predictions.npy       # Patch predictions
│   ├── patch_targets.npy           # Ground truth (Patch eval)
│   ├── predictions.npy             # Default predictions
│   └── targets.npy                 # Default ground truth
│
├── summaries/                      # Summary reports
│   ├── eff_results.json            # EfficientNet metrics (JSON)
│   ├── eff_results.npy             # EfficientNet metrics (NumPy)
│   ├── fusion_results.json         # Fusion metrics (JSON)
│   ├── fusion_results.npy          # Fusion metrics (NumPy)
│   ├── patch_results.json          # Patch metrics (JSON)
│   ├── patch_results.npy           # Patch metrics (NumPy)
│   ├── results.json                # Default results (JSON)
│   └── results.npy                 # Default results (NumPy)
│
├── Eff_Results/                    # Archived EfficientNet results
├── Fusion_Results/                 # Archived Fusion model results
├── Patch_Results/                  # Archived Patch CNN results
│
└── README.md                       # This file
```

## File Descriptions

### models/
All `.pth` files are PyTorch state dictionaries containing trained model weights.
- **best_model.pth**: Best checkpoint from training
- **eff_model.pth**: EfficientNet B0 backbone model
- **fusion_model.pth**: Fusion of EfficientNet + Patch CNN
- **patch_model.pth**: Custom Patch CNN model

### metrics/
Per-class evaluation metrics (shape: num_classes = 7).
- `.npy` files contain NumPy arrays with accuracy/F1 scores per class

### thresholds/
Optimal decision thresholds for multi-label classification (shape: num_classes = 7).
- Values are clipped to [0.2, 0.8] to prevent model collapse

### predictions/
Predictions and ground truth labels from model evaluation.
- Predictions are probabilities (0-1)
- Targets are binary labels

### summaries/
Aggregate metrics and evaluation results in both JSON and NumPy formats.
- JSON format is human-readable
- NumPy format is for programmatic access

---

## Model Performance Summary

| Model | Models | Metrics | Thresholds | Predictions |
|-------|--------|---------|-----------|-------------|
| **EfficientNet** | ✅ eff_model.pth | ✅ 2 files | ✅ thresholds | ✅ 2 files |
| **Patch CNN** | ✅ patch_model.pth | ✅ 2 files | ✅ thresholds | ✅ 2 files |
| **Fusion** | ✅ fusion_model.pth | ✅ 2 files | ✅ thresholds | No predictions |
| **Default** | ✅ best_model.pth | ✅ 2 files | — | ✅ 2 files |

---

## Usage

Files are generated automatically when running:

```bash
python train.py   # Creates models/ and thresholds/
python test.py    # Creates metrics/ and summaries/
```

File paths are configured in `configs/config.py`.

## Loading Results

### In Python
```python
import numpy as np
import torch
import json

# Load model
model = torch.load('results/models/fusion_model.pth')

# Load metrics
acc = np.load('results/metrics/fusion_per_class_acc.npy')
f1 = np.load('results/metrics/fusion_per_class_f1.npy')

# Load thresholds
thresholds = np.load('results/thresholds/fusion_thresholds.npy')

# Load summary
with open('results/summaries/fusion_results.json') as f:
    results = json.load(f)
```
