import torch
from torch import nn
from typing import Callable, List, Type

class BaseConstrs(nn.Module):
    def __init__(self, T: int = 4, 
                 L: int = 4, 
                 vthr: float = 8.0, 
                 tau: float = 1.0, 
                 regularizer: Type[nn.Module] = None, 
                 momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([vthr]), requires_grad=False)
        self.regularizer = regularizer
        self.T = T
        self.tau = tau
        self.momentum = momentum
        self.relu = nn.ReLU(inplace=True)

    def constraints(self,x):
        if self.training:
            vthr = (1-self.momentum)*torch.max(x.detach())+self.momentum*self.vthr
            self.vthr.copy_(vthr)
        x = self.relu(x)
        return x

    def forward(self, x):
        if self.regularizer is not None:
            loss = self.regularizer(x.clone())
        x = self.constraints(x)
        return x 