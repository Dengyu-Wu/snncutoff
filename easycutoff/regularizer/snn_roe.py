import torch
from torch import nn
import numpy as np


class SNNROE(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_loss = True

    def forward(self,x,mem):
        rank = len(x.size())-2  # N,T,C,W,H
        dim = -np.arange(rank)-1
        dim = list(dim)
        x_clone = (x+mem)
        x_norm = (x_clone.pow(2).sum(dim=dim)+1e-5)**0.5
        cs = x_norm[1:,...]/x_norm[:-1,...]
        cs = cs.mean(0)
        r = 1/torch.min(cs)

        r = torch.maximum(r,torch.tensor(1.0))
        loss = torch.log(r)       
        return loss

# class SNNROE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.add_loss = True

#     def forward(self,x,mem):
#         rank = len(x.size())-2  # N,T,C,W,H
#         dim = -np.arange(rank)-1
#         dim = list(dim)
#         x_clone = (x+mem).sum(0)
#         x_clone = torch.maximum(x_clone,torch.tensor(0.0))
#         xmax = torch.max(x_clone)
#         sigma = (x_clone.pow(2).mean(dim=dim)+1e-5)**0.5
#         r = xmax/torch.min(sigma)

#         r = torch.maximum(r,torch.tensor(1.0))
#         loss = torch.log(r)       
#         return loss



