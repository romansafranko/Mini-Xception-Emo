"""
Utility classes:
* FocalLoss        – tackles class imbalance by down‑weighting easy samples
* SubsetWithTransform – attach a transform to a torch.utils.data.Subset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class FocalLoss(nn.Module):
    r"""
    Focal loss (Lin et al., 2017).

    FL(p_t) = -α · (1 − p_t)^γ · log(p_t)

    Parameters
    ----------
    alpha : float
        Balancing factor between classes.
    gamma : float
        Focusing parameter to reduce the loss contribution from easy samples.
    reduction : {"mean", "sum", "none"}
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SubsetWithTransform(Dataset):
    """
    A thin wrapper that lets you retrofit a *transform* onto a Subset object.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
