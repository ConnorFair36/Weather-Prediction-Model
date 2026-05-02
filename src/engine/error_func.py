import torch
from torch import nn
import torch.nn.functional as F

class WeightedMSELoss(nn.Module):
    def __init__(self, rain_weight=10.0):
        super().__init__()
        self.rain_weight = rain_weight

    def forward(self, pred, target):
        weights = torch.where(target > 0, self.rain_weight, 1.0)
        return (weights * (pred - target) ** 2).mean()

class WeightedAVELoss(nn.Module):
    def __init__(self, rain_weight=10.0):
        super().__init__()
        self.rain_weight = rain_weight

    def forward(self, pred, target):
        weights = torch.where(target > 0, self.rain_weight, 1.0)
        return (weights * torch.abs(pred - target)).mean()


# Source: https://www.tandfonline.com/doi/full/10.1080/17538947.2023.2253206#d1e1020 Section 3.4.2
#   I am using weighted MSE because I expect the same issues that I had with MSE to show up again due to the small values in 
#    the rain being predicted
class SpatialLoss(nn.Module):
    def __init__(self, rain_weight):
        super().__init__()
        # Sobel Kernels
        self.register_buffer('kernel_x', torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('kernel_y', torch.tensor(
            [[ 1.,  2.,  1.], [ 0.,  0.,  0.], [-1., -2., -1.]]).view(1, 1, 3, 3))
        self.error = WeightedMSELoss(rain_weight)

    def forward(self, pred, target):
        B, T, W, H = pred.shape
        # Expand kernels to (T, 1, 3, 3) for grouped conv over T channels
        kx = self.kernel_x.repeat(T, 1, 1, 1)
        ky = self.kernel_y.repeat(T, 1, 1, 1)

        pred_x   = F.conv2d(pred,   kx, padding=1, groups=T)
        pred_y   = F.conv2d(pred,   ky, padding=1, groups=T)
        target_x = F.conv2d(target, kx, padding=1, groups=T)
        target_y = F.conv2d(target, ky, padding=1, groups=T)

        return self.error(pred_x, target_x) + self.error(pred_y, target_y)

# Source: https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0 Section 1.3
class LocalLoss(nn.Module):
    """Combines the weighted mse (value loss) with the spatial loss and scales each with their own reciprical."""
    def __init__(self,  rain_weight=10.0, spatial_rain_weight=10.0):
        super().__init__()
        self.value_loss = WeightedMSELoss(rain_weight)
        self.spatial_loss = SpatialLoss(spatial_rain_weight)

        self.last_value_loss = 0
        self.last_spatial_loss = 0
    
    def forward(self, pred, target):
        this_value_loss = self.value_loss(pred, target) 
        this_spatial_loss = self.spatial_loss(pred, target)
        self.last_value_loss = this_value_loss.detach()
        self.last_spatial_loss = this_spatial_loss.detach()
        return (this_value_loss / this_value_loss.detach()) + (this_spatial_loss / this_spatial_loss.detach())

class LocalLossL1(nn.Module):
    """Combines the weighted mse (value loss) with the spatial loss and scales each with their own reciprical."""
    def __init__(self,  rain_weight=10.0, spatial_rain_weight=10.0):
        super().__init__()
        self.value_loss = WeightedAVELoss(rain_weight)
        self.spatial_loss = SpatialLoss(spatial_rain_weight)

        self.last_value_loss = 0
        self.last_spatial_loss = 0
    
    def forward(self, pred, target):
        this_value_loss = self.value_loss(pred, target) 
        this_spatial_loss = self.spatial_loss(pred, target)
        self.last_value_loss = this_value_loss.detach()
        self.last_spatial_loss = this_spatial_loss.detach()
        return (this_value_loss / this_value_loss.detach()) + (this_spatial_loss / this_spatial_loss.detach())
