import torch
from torch import nn

class WeightedMSELoss(nn.Module):
    def __init__(self, rain_weight=10.0):
        super().__init__()
        self.rain_weight = rain_weight

    def forward(self, pred, target):
        weights = torch.where(target > 0, self.rain_weight, 1.0)
        return (weights * (pred - target) ** 2).mean()