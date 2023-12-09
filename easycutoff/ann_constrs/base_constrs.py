import torch
from torch import nn
from torch.autograd import Function
from easycutoff.regularizer import ROE
from typing import Callable, List, Type

class BaseConstrs(nn.Module):
    def __init__(self, T: int = 4, L: int = 4, vthr: float = 8.0, regularizer: Type[ROE] = None, momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([vthr]), requires_grad=False)
        self.regularizer = regularizer
        self.T = T
        self.momentum = momentum
       
    def constraints(self,x):
        if self.training:
            vthr = (1-self.momentum)*torch.max(x.detach())+self.momentum*self.vthr
            self.vthr.copy_(vthr)
        x = nn.ReLU()(x)
        return x

    def reshape(self,x):
        batch_size = int(x.shape[0]/self.T)
        new_dim = [self.T, batch_size]
        new_dim.extend(x.shape[1:])
        return x.reshape(new_dim)

    def forward(self, x):
        x = self.constraints(x)
        x = self.reshape(x)
        if self.regularizer is not None:
            loss = self.regularizer(x)
        return x 