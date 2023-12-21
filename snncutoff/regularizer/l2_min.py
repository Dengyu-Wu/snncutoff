import torch
from torch import nn
import numpy as np

class L2Min(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_loss = True
        
    def forward(self,x):
        rank = len(x.size())-2  # N,T,C,W,H
        dim = -np.arange(rank)-1
        dim = list(dim)
        x_clone = torch.maximum(x,torch.tensor(0.0))
        sigma = (x_clone.pow(2).mean(dim=dim)+1e-5)**0.5
        r = torch.min(sigma)

        loss = 1/r   
        return loss


