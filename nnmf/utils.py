import torch
import torch.nn as nn
import torch.nn.functional as F
from nnmf import DEBUG
if DEBUG:
    import debug.functional as F

class PowerSoftmax(nn.Module):
    def __init__(self, power, dim):
        super().__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        if self.power == 1:
            return F.normalize(x, p=1, dim=self.dim)
        power_x = torch.pow(x, self.power)
        return power_x / torch.sum(power_x, dim=self.dim, keepdim=True)