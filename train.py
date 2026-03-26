import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from configs.config import *
from data.dataset import CattleDataset
from models.dual_model import DualModel
from utils.augmentations import *
from utils.mixup import mixup
from utils.loss import FocalLoss
from utils.threshold import find_best_thresholds


# =========================
# CLASS WEIGHTS
# =========================
def compute_class_weights(loader):
    total = 0
    pos_counts = torch.zeros(NUM_CLASSES)

    for _, y in loader:
        pos_counts += y.sum(dim=0)
        total += y.size(0)

    weights = total / (pos_counts + 1e-6)
    weights = weights / weights.sum()

    return weights.to(DEVICE)


# =========================
# EVALUATE
# =========================
def evaluate(model, loader):
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(y)

    return torch.cat(all_probs).numpy(), torch.cat(all_targets).numpy()


# =========================
# TRAIN
# =========================
def train():
    print("🔹 Training Fusion Model")

    train_ds = CattleDataset(BASE_DIR, TRAIN_EXCEL, get_train_transforms())
    val_ds   = CattleDataset(BASE_DIR, VAL_EXCEL, get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DualModel(NUM_CLASSES, mode="fusion").to(DEVICE)

    # ⚠️ IMPORTANT CHANGE
    # Fusion model → use BCEWithLogitsLoss (more stable than focal here)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()

            x, y = mixup(x, y)

            optimizer.zero_grad()
            out = model(x)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).float()

                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), FUSION_MODEL_PATH)
            print("✅ Best model saved")

    print(f"\n🏆 Best epoch: {best_epoch} | Val Loss: {best_val_loss:.4f}")

    # =========================
    # THRESHOLD TUNING
    # =========================
    model.load_state_dict(torch.load(FUSION_MODEL_PATH))

    val_probs, val_targets = evaluate(model, val_loader)

    thresholds = find_best_thresholds(val_targets, val_probs)

    # 🔥 IMPORTANT FIX (prevents Patch-like collapse)
    thresholds = np.clip(thresholds, 0.2, 0.8)

    np.save(FUSION_THRESHOLD_PATH, thresholds)

    print("✅ Fusion thresholds saved")


if __name__ == "__main__":
    train()