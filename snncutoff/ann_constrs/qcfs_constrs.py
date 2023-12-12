import torch
from torch import nn
from torch.autograd import Function
from .base_constrs import BaseConstrs
from snncutoff.regularizer import ROE
from typing import Callable, List, Type



class QCFSConstrs(BaseConstrs):
    def __init__(self, T: int = 4, L: int = 4, vthr: float = 8.0, regularizer: Type[ROE] = None, momentum=0.9):
        super().__init__()
        self.vthr = nn.Parameter(torch.tensor([vthr]), requires_grad=True)
        self.T = T
        self.L = L
        self.relu = nn.ReLU()
        self.regularizer = regularizer
        
    def constraints(self, x):
        x = self.relu(x)
        x = x / self.vthr
        x = myfloor(x*self.L+0.5)/self.L
        x = torch.clamp(x, 0, 1)
        x = x * self.vthr
        return x


class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

# class TCL(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
#     def forward(self, x):
#         x = F.relu(x, inplace='True')
#         x = self.up - x
#         x = F.relu(x, inplace='True')
#         x = self.up - x
#         return x
