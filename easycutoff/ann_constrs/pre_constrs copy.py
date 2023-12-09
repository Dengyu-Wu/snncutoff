import torch
from torch import nn
from torch.autograd import Function
from easycutoff.regularizer import ROE
from typing import Callable, List, Type
from .base_constrs import BaseConstrs

class PostConstrs(BaseConstrs):
    def __init__(self, T: int = 4, vthr: float = 8.0, module: Type[nn.Module] = None, momentum=0.9):
        super().__init__()
        self.T = T
        self.module = module
       
    def reshape(self,x):
        new_dim = [int(x.shape[1]*self.T),]
        new_dim.extend(x.shape[2:])
        print(new_dim)
        return x.reshape(new_dim)

    def forward(self, x):
        x = self.reshape(x)
        x = self.module(x)
        return x 