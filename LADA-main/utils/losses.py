import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, target, sample_weights=None):
        loss_per_sample = F.cross_entropy(logit, target, reduction='none')
        if sample_weights is not None:
            return (loss_per_sample * sample_weights).mean()
        else:
            return loss_per_sample.mean()


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, logit, target):
        return focal_loss(F.cross_entropy(logit, target, reduction='none', weight=self.weight), self.gamma)
