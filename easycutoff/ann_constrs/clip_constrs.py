import torch
from torch import nn
from .base_constrs import BaseConstrs


class ClipConstrs(BaseConstrs):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        self.relu = nn.ReLU()
    def constraints(self, x):
        x = self.relu(x)
        x = x / self.up
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

