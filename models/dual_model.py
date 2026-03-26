import torch
import torch.nn as nn
import torchvision.models as models

# =========================
# EfficientNet Backbone
# =========================
class EfficientNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier = nn.Identity()  # remove final FC

    def forward(self, x):
        return self.model(x)   # output: (B, 1280)


# =========================
# Patch CNN (Local Features)
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
# Attention Fusion
# =========================
class AttentionFusion(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        
        self.fusion_dim = dim1  # Use dim1 as the common dimension

        # Project dim2 to dim1
        self.proj = nn.Linear(dim2, dim1)

        # Attention weight network
        self.fc = nn.Sequential(
            nn.Linear(dim1 + dim1, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, f1, f2):
        # Project f2 to match f1's dimension
        f2_proj = self.proj(f2)
        
        # Compute attention weights
        weights = self.fc(torch.cat([f1, f2_proj], dim=1))
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)

        # Fuse using attention weights
        return w1 * f1 + w2 * f2_proj


# =========================
# Dual Model (Flexible)
# =========================
class DualModel(nn.Module):
    def __init__(self, num_classes, mode="fusion"):
        super().__init__()

        self.mode = mode

        # Backbones
        self.eff = EfficientNetBackbone()
        self.patch = PatchCNN()

        # Feature dimensions
        self.eff_dim = 1280
        self.patch_dim = 128

        # BatchNorm (important)
        self.bn1 = nn.BatchNorm1d(self.eff_dim)
        self.bn2 = nn.BatchNorm1d(self.patch_dim)

        # Attention fusion
        self.attn = AttentionFusion(self.eff_dim, self.patch_dim)

        # Classifiers based on mode
        if self.mode == "eff":
            self.fc = nn.Linear(self.eff_dim, num_classes)

        elif self.mode == "patch":
            self.fc = nn.Linear(self.patch_dim, num_classes)

        elif self.mode == "fusion":
            self.fc = nn.Linear(self.eff_dim, num_classes)

        else:
            raise ValueError("Mode must be 'eff', 'patch', or 'fusion'")

    def forward(self, x):

        # Extract features
        f1 = self.bn1(self.eff(x))     # (B, 1280)
        f2 = self.bn2(self.patch(x))   # (B, 128)

        # =========================
        # Mode Handling
        # =========================
        if self.mode == "eff":
            return self.fc(f1)

        elif self.mode == "patch":
            return self.fc(f2)

        elif self.mode == "fusion":
            fused = self.attn(f1, f2)
            return self.fc(fused)