import torch
from torch import nn
from torch.autograd import Function
from snncutoff.regularizer import ROE
from typing import Callable, List, Type
from .base_constrs import BaseConstrs

class PostConstrs(BaseConstrs):
    def __init__(self, T: int = 4, vthr: float = 8.0, module: Type[nn.Module] = None, momentum=0.9):
        super().__init__()
        self.T = T
        self.module = module

    def forward(self, x):
        if self.module is not None:
            x = self.module(x)
        x = self.reshape(x)
        return x 