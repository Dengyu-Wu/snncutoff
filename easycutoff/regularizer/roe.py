import torch
from torch import nn
import numpy as np

class ROE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        rank = len(x.size())-1   # N,T,C,W,H
        rank = np.clip(rank,3,rank)
        dim = np.arange(rank-1)+2   
        dim = list(dim)
        x_clone = torch.maximum(x.clone(),torch.tensor(0.0))
        xmax = torch.max(x_clone)
        sigma = (x_clone.pow(2).mean(dim=dim)+1e-5)**0.5
        r = xmax/torch.min(sigma)
        
        r = torch.maximum(r,torch.tensor(1.0))
        loss = torch.log(r)       
        return loss


