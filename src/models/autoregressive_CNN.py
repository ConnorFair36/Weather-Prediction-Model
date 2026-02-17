import numpy as np
import pandas as pd
import torch
from torch import nn, optim

class AR_CNN(nn.Module):
    """
    Represents an autoregressive CNN that predicts the current hour rainfall based on the previous hour data.

    :param filters1: the number of filters for the first layer.
    :param filters2: the number of filters for the second layer.
    """
    def __init__(self, filters1=16, filters2=64):
        super(AR_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, filters1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters1, filters2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters2, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.tensor):
        if x.ndim == 5:
            x = x.squeeze(dim=1)
        return self.conv_layers(x)