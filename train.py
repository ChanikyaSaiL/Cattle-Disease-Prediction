import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.config import *
from data.dataset import CattleDataset
from models.dual_model import DualModel
from utils.augmentations import *
from utils.mixup import mixup
from utils.loss import FocalLoss


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
# TRAIN
# =========================
def train():
    print("🔥 Training Fusion Model with Class-Aware Attention + Learnable Thresholds")

    train_ds = CattleDataset(BASE_DIR, TRAIN_EXCEL, get_train_transforms())
    val_ds   = CattleDataset(BASE_DIR, VAL_EXCEL, get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = DualModel(NUM_CLASSES, mode="fusion").to(DEVICE)

    class_weights = compute_class_weights(train_loader)
    criterion = FocalLoss(alpha=class_weights, gamma=2)

    # 🔥 IMPORTANT: include thresholds in optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()

            x, y = mixup(x, y)

            optimizer.zero_grad()

            logits, thresholds = model(x)

            loss = criterion(logits, y)
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

                logits, _ = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # SAVE BEST MODEL
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "fusion_q1_model.pth")
            print("✅ Best model saved")

    print("🎯 Training Complete")


if __name__ == "__main__":
    train()