import torch
from torch import nn
from torch.autograd import Function
from snncutoff.regularizer import ROE
from typing import Callable, List, Type
from .base_constrs import BaseConstrs

class PreConstrs(BaseConstrs):
    def __init__(self, T: int = 4, vthr: float = 8.0, module: Type[nn.Module] = None, momentum=0.9, multistep: bool = True):
        super().__init__()
        self.T = T
        self.module = module
        self.multistep = multistep
       
    def reshape(self,x):
        if self.multistep:
            new_dim = [int(x.shape[1]*self.T),]
            new_dim.extend(x.shape[2:])
            return x.reshape(new_dim)
        else: 
            return x[0]
        
    def forward(self, x):
        x = self.reshape(x)
        x = self.module(x)
        return x 