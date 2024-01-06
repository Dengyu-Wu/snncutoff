import torch
from torch import nn
import numpy as np


class RCSSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_loss = True

    def forward(self,x,mem):
        rank = len(x.size())-2  # T,N,C,W,H
        dim = -np.arange(rank)-1
        dim = list(dim)
        r_t = x+torch.relu(mem)
        r_d = r_t.mean(0,keepdim=True).detach()
        r_d_norm = (r_d.pow(2).sum(dim=dim)+1e-5)**0.5
        r_t_norm = (r_t.pow(2).sum(dim=dim)+1e-5)**0.5
        cs = (r_t*r_d).sum(dim=dim)/(r_t_norm*r_d_norm)
        cs = cs.mean(0,keepdim=True)
        cs = 1/(cs+1e-5)
        return  cs