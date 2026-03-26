import numpy as np
import torch
from configs.config import DEVICE

def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(DEVICE)

    x_mix = lam * x + (1 - lam) * x[index]
    y_mix = lam * y + (1 - lam) * y[index]

    return x_mix, y_mix