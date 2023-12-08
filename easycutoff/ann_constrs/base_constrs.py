import torch
from torch import nn
from torch.autograd import Function
from easycutoff.regularizer import ROE
from typing import Callable, List, Type

class BaseConstrs(nn.Module):
    def __init__(self, T=4, regularizer: Type[ROE] = None):
        super().__init__()
        self.relu = nn.ReLU()
        self.regularizer = regularizer
        self.T = T
       
    def constraints(self,x):
        x = self.relu(x)
        return x


    def forward(self, x):
        x = self.constraints(x)
        if self.regularizer is not None:
            loss = self.regularizer(x)
        return x 