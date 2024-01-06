import torch
from torch import nn
from torch.autograd import Function
from .base_constrs import BaseConstrs
from typing import Callable, List, Type
from snncutoff.gradients import GradFloor

class QCFSConstrs(BaseConstrs):
    def __init__(self, T: int = 4, 
                 L: int = 4, 
                 vthr: float = 8.0, 
                 tau: float = 1.0, 
                 gradient: Type[Function] = GradFloor, 
                 regularizer: Type[nn.Module] = None, momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([vthr]), requires_grad=True)
        self.regularizer = regularizer
        self.T = T
        self.L = L
        self.tau = tau
        self.momentum = momentum
        self.gradient = gradient.apply
        self.relu = nn.ReLU(inplace=True)
        
    def constraints(self, x):
        x = self.relu(x)
        x = x / self.vthr
        x = self.gradient(x*self.L+0.5)/self.L
        x = torch.clamp(x, 0, 1)
        x = x * self.vthr
        return x
