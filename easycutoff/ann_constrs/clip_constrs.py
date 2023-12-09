import torch
from torch import nn
from .base_constrs import BaseConstrs
from typing import Callable, List, Type
from easycutoff.regularizer import ROE


class ClipConstrs(BaseConstrs):
    def __init__(self, T: int = 4, L: int = 4, vthr: float = 8.0, regularizer: Type[ROE] = None, momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([vthr]), requires_grad=True)
        self.T = T
        self.relu = nn.ReLU()
        self.regularizer = regularizer

    def constraints(self, x):
        x = self.relu(x)
        x = x / self.vthr
        x = torch.clamp(x, 0, 1)
        x = x * self.vthr
        return x

