import torch
from torch.utils.data import DataLoader
import numpy as np
import json

from sklearn.metrics import (
    f1_score,
    hamming_loss,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from configs.config import *
from data.dataset import CattleDataset
from models.dual_model import DualModel
from utils.augmentations import get_val_transforms


def test():
    print("🔹 Testing Fusion Model")

    test_ds = CattleDataset(BASE_DIR, TEST_EXCEL, get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DualModel(NUM_CLASSES, mode="fusion").to(DEVICE)
    model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=DEVICE))
    model.eval()

    thresholds = np.load(FUSION_THRESHOLD_PATH)

    all_probs, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(y)

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    preds = (all_probs > thresholds).astype(int)

    # =========================
    # METRICS
    # =========================
    subset_acc = (preds == all_targets).all(axis=1).mean()

    f1_micro = f1_score(all_targets, preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, preds, average='macro', zero_division=0)

    active_classes = (all_targets.sum(axis=0) > 0)

    f1_macro_active = f1_score(
        all_targets[:, active_classes],
        preds[:, active_classes],
        average='macro',
        zero_division=0
    )

    hamming = hamming_loss(all_targets, preds)

    per_class_acc = (preds == all_targets).mean(axis=0)
    per_class_f1 = f1_score(all_targets, preds, average=None, zero_division=0)

    try:
        roc_auc = roc_auc_score(all_targets, all_probs, average='macro')
    except:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(all_targets, all_probs, average='macro')
    except:
        pr_auc = float("nan")

    # =========================
    # PRINT
    # =========================
    print("\n📊 Evaluation Metrics")
    print("=" * 60)

    print(f"Subset Accuracy        : {subset_acc:.4f}")
    print(f"F1 Micro              : {f1_micro:.4f}")
    print(f"F1 Macro (All)        : {f1_macro:.4f}")
    print(f"F1 Macro (Active)     : {f1_macro_active:.4f}")
    print(f"Hamming Loss          : {hamming:.4f}")
    print(f"ROC-AUC (Macro)       : {roc_auc:.4f}")
    print(f"PR-AUC (Macro)        : {pr_auc:.4f}")

    print("\n📌 Per-Class Metrics:")
    for i in range(NUM_CLASSES):
        print(f"Class {i} → Acc: {per_class_acc[i]:.4f}, F1: {per_class_f1[i]:.4f}")

    print("\n📌 Classification Report:")
    print(classification_report(all_targets, preds, zero_division=0))

    # =========================
    # SAVE
    # =========================
    results = {
        "subset_accuracy": float(subset_acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_macro_active": float(f1_macro_active),
        "hamming_loss": float(hamming),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    np.save("results/summaries/fusion_results.npy", results)
    np.save("results/metrics/fusion_per_class_acc.npy", per_class_acc)
    np.save("results/metrics/fusion_per_class_f1.npy", per_class_f1)

    with open("results/summaries/fusion_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Fusion metrics saved")


if __name__ == "__main__":
    test()