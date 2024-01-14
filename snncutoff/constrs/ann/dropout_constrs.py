import torch
from torch import nn
from typing import Callable, List, Type

class DropoutConstrs(nn.Module):
    def __init__(self, 
                 module: Type[nn.Module] = None, 
                 p: float = 0.0,):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.module(x)
        x = self.dropout(x)
        return x 