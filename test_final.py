import torch
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import (
    f1_score,
    hamming_loss,
    classification_report
)

from configs.config import *
from data.dataset import CattleDataset
from models.dual_model import DualModel
from utils.augmentations import get_val_transforms


# =========================
# TEST FUNCTION
# =========================
def test():

    for mode in ["eff", "patch", "fusion"]:

        print(f"\n🚀 Testing Mode: {mode}")

        # =========================
        # DATA
        # =========================
        test_ds = CattleDataset(BASE_DIR, TEST_EXCEL, get_val_transforms())
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # =========================
        # MODEL
        # =========================
        model = DualModel(NUM_CLASSES, mode=mode).to(DEVICE)
        model.load_state_dict(torch.load(f"{mode}_model_final.pth", map_location=DEVICE))
        model.eval()

        all_probs = []
        all_targets = []

        # =========================
        # INFERENCE
        # =========================
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)

                logits = model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_targets.append(y)

        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # 🔥 FIXED THRESHOLD
        preds = (all_probs > 0.5).astype(int)

        # =========================
        # METRICS
        # =========================
        subset_acc = (preds == all_targets).all(axis=1).mean()

        f1_micro = f1_score(all_targets, preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_targets, preds, average='macro', zero_division=0)

        hamming = hamming_loss(all_targets, preds)

        print("\n📊 Results")
        print("=" * 50)
        print(f"Subset Accuracy : {subset_acc:.4f}")
        print(f"F1 Micro        : {f1_micro:.4f}")
        print(f"F1 Macro        : {f1_macro:.4f}")
        print(f"Hamming Loss    : {hamming:.4f}")

        print("\n📌 Classification Report:")
        print(classification_report(all_targets, preds, zero_division=0))


if __name__ == "__main__":
    test()