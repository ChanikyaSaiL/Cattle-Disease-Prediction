import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# EfficientNet Backbone
# =========================
class EfficientNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights="DEFAULT")
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)   # (B, 1280)


# =========================
# Patch CNN
# =========================
class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)  # (B, 128)


# =========================
# Class-Aware Fusion
# =========================
class ClassAwareFusion(nn.Module):
    def __init__(self, eff_dim, patch_dim, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.proj = nn.Linear(patch_dim, eff_dim)

        self.attn = nn.Sequential(
            nn.Linear(eff_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes * 2)
        )

    def forward(self, f1, f2):

        f2_proj = self.proj(f2)  # (B, 1280)

        B = f1.size(0)

        combined = torch.cat([f1, f2_proj], dim=1)
        weights = self.attn(combined)

        weights = weights.view(B, self.num_classes, 2)
        weights = torch.softmax(weights, dim=2)

        w1 = weights[:, :, 0].unsqueeze(-1)
        w2 = weights[:, :, 1].unsqueeze(-1)

        f1_exp = f1.unsqueeze(1)
        f2_exp = f2_proj.unsqueeze(1)

        fused = w1 * f1_exp + w2 * f2_exp  # (B, num_classes, 1280)

        return fused


# =========================
# FINAL MODEL
# =========================
class DualModel(nn.Module):
    def __init__(self, num_classes, mode="fusion"):
        super().__init__()

        self.mode = mode
        self.num_classes = num_classes

        self.eff = EfficientNetBackbone()
        self.patch = PatchCNN()

        self.eff_dim = 1280
        self.patch_dim = 128

        self.bn1 = nn.BatchNorm1d(self.eff_dim)
        self.bn2 = nn.BatchNorm1d(self.patch_dim)

        self.fusion = ClassAwareFusion(self.eff_dim, self.patch_dim, num_classes)

        if mode == "eff":
            self.fc = nn.Linear(self.eff_dim, num_classes)

        elif mode == "patch":
            self.fc = nn.Linear(self.patch_dim, num_classes)

        elif mode == "fusion":
            self.fc = nn.Linear(self.eff_dim, 1)

        else:
            raise ValueError("Mode must be 'eff', 'patch', or 'fusion'")

    def forward(self, x):

        f1 = self.bn1(self.eff(x))
        f2 = self.bn2(self.patch(x))

        if self.mode == "eff":
            return self.fc(f1)

        elif self.mode == "patch":
            return self.fc(f2)

        elif self.mode == "fusion":
            fused = self.fusion(f1, f2)

            # 🔥 CRITICAL: Residual Stabilization
            fused = 0.7 * f1.unsqueeze(1) + 0.3 * fused

            logits = self.fc(fused).squeeze(-1)

            return logits