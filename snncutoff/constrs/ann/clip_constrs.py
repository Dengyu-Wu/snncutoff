import torch
from torch import nn
from .base_constrs import BaseConstrs
from typing import Callable, List, Type


class ClipConstrs(BaseConstrs):
    def __init__(self, T: int = 4, 
                 L: int = 4, 
                 vthr: float = 8.0, 
                 tau: float = 1.0, 
                 regularizer: Type[nn.Module] = None, 
                 momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([vthr]), requires_grad=True)
        self.regularizer = regularizer
        self.T = T
        self.tau = tau
        self.momentum = momentum
        self.relu = nn.ReLU(inplace=True)

    def constraints(self, x):
        # if self.training:
        #     xmax = torch.mean(x.detach().abs())*100
        #     vthr = (1-self.momentum)*xmax+self.momentum*self.vthr
        #     self.vthr.copy_(vthr)
        x = self.relu(x)
        x = x / self.vthr
        x = torch.clamp(x, 0, 1)
        x = x * self.vthr
        return x

