import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)

        if self.alpha is not None:
            alpha = self.alpha.unsqueeze(0)
            bce_loss = alpha * bce_loss

        loss = ((1 - pt) ** self.gamma) * bce_loss
        return loss.mean()