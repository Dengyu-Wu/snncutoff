import torch
from torch import nn
from typing import Callable, List, Type
from .base_constrs import BaseConstrs

class PostConstrs(BaseConstrs):
    def __init__(self, T: int = 4, vthr: float = 8.0, module: Type[nn.Module] = None, momentum=0.9, multistep: bool = True):
        super().__init__()
        self.T = T
        self.module = module
        self.multistep = multistep

    def forward(self, x):
        if self.module is not None:
            x = self.module(x)
        if self.multistep:
            return self.reshape(x)
        else: 
            return x.unsqueeze(0)