"""
Utility classes: FocalLoss and dataset‑wrapper with on‑the‑fly transform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017) for handling class imbalance.

    Args
    ----
    alpha : balancing factor between positive/negative examples
    gamma : focusing parameter (γ = 2 is common)
    reduction : 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.reduction  = reduction

    def forward(self, inputs, targets):
        """
        inputs  — un‑normalized logits, shape [N, C]
        targets — ground‑truth class indices, shape [N]
        """
        ce_loss  = F.cross_entropy(inputs, targets, reduction='none')
        pt       = torch.exp(-ce_loss)           # probability of the true class
        focal    = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal)
        if self.reduction == 'sum':
            return torch.sum(focal)
        return focal                             # 'none'


class SubsetWithTransform(Dataset):
    """
    Wrap a torch.utils.data.Subset and lazily apply a transform.

    Useful because torchvision’s ImageFolder returns PIL images; we
    want different transforms for train/val but keep a single subset
    object produced by **random_split**.
    """
    def __init__(self, subset, transform=None):
        self.subset   = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]          # original (image, label)
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
