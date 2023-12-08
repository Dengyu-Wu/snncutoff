import torch
from torch import nn
from torch.autograd import Function
from .base_constrs import BaseConstrs
from easycutoff.regularizer import ROE
from typing import Callable, List, Type



class QCFSConstrs(BaseConstrs):
    def __init__(self, thresh=8., T=32, regularizer: Type[ROE] = None):
        BaseConstrs.__init__(self)
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.T = T
        self.relu = nn.ReLU()
        self.regularizer = regularizer
        
    def constraints(self, x):
        x = self.relu(x)
        x = x / self.thresh
        x = myfloor(x*self.T+0.5)/self.T
        x = torch.clamp(x, 0, 1)
        x = x * self.thresh
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
