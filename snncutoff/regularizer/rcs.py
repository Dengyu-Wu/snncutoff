import torch
from torch import nn
import numpy as np


class RCS(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_loss = True
        
    def forward(self,x):
        rank = len(x.size())-2  # N,T,C,W,H
        dim = -np.arange(rank)-1
        dim = list(dim)
        x_clone = torch.maximum(x,torch.tensor(0.0))
        xmax = torch.max(x_clone)
        sigma = (x_clone.pow(2).mean(dim=dim)+1e-5)**0.5
        r = xmax/sigma
        return r


