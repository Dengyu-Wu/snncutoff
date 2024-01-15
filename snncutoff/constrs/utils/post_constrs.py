import torch
from torch import nn
from typing import Type

class PostConstrs(nn.Module):
    def __init__(self, T: int = 4, 
                 module: Type[nn.Module] = None, 
                 multistep: bool = True):
        super().__init__()
        self.T = T
        self.module = module
        self.multistep = multistep

    def reshape(self,x):
        batch_size = int(x.shape[0]/self.T)
        new_dim = [self.T, batch_size]
        new_dim.extend(x.shape[1:])
        return x.reshape(new_dim)

    def forward(self, x):
        if self.module is not None:
            x = self.module(x)
        if self.multistep:
            return self.reshape(x)
        else: 
            return x.unsqueeze(0)