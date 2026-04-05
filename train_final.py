import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from configs.config import *
from data.dataset import CattleDataset
from models.dual_model import DualModel
from utils.augmentations import *
from utils.mixup import mixup
from utils.loss import FocalLoss


# =========================
# COMPUTE CLASS WEIGHTS
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
# CREATE WEIGHTED SAMPLER
# =========================
def create_sampler(dataset):
    targets = [y.numpy() for _, y in dataset]
    targets = np.array(targets)

    class_counts = targets.sum(axis=0)
    weights = 1.0 / (class_counts + 1e-6)

    sample_weights = (targets * weights).sum(axis=1)

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return sampler


# =========================
# TRAIN FUNCTION
# =========================
def train():

    for mode in ["eff", "patch", "fusion"]:

        print(f"\n🚀 Training Mode: {mode}")

        # =========================
        # DATA
        # =========================
        train_ds = CattleDataset(BASE_DIR, TRAIN_EXCEL, get_train_transforms())
        val_ds   = CattleDataset(BASE_DIR, VAL_EXCEL, get_val_transforms())

        sampler = create_sampler(train_ds)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # =========================
        # MODEL
        # =========================
        model = DualModel(NUM_CLASSES, mode=mode).to(DEVICE)

        # =========================
        # LOSS
        # =========================
        class_weights = compute_class_weights(train_loader)
        criterion = FocalLoss(alpha=class_weights, gamma=3)  # 🔥 stronger focus

        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_val_loss = float("inf")

        # =========================
        # TRAIN LOOP
        # =========================
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0

            for x, y in train_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).float()

                # 🔥 Controlled MixUp (only for EfficientNet)
                if mode == "eff" and np.random.rand() < 0.3:
                    x, y = mixup(x, y)

                optimizer.zero_grad()

                logits = model(x)
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

                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # SAVE BEST
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{mode}_model_final.pth")
                print("✅ Best model saved")

        print(f"🏆 Finished training {mode}")


if __name__ == "__main__":
    train()