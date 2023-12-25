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
        x_clone = torch.relu(x)
        # x_norm = (x_clone.pow(2).sum(dim=dim)+1e-5)**0.5
        # cs = (x_clone[1:,...]*x_clone[-1:,...]).sum(dim=dim)/(x_norm[1:,...]*x_norm[:-1,...])
        r = []
        for t in range(x_clone.size()[0]):
            t += 1
            r_t = x_clone[0:t].mean(0)
            r.append(r_t)
        r = torch.stack(r, dim=0)
        # x_norm = (r.pow(2).sum(dim=dim)+1e-5)**0.5
        # cs = (r[1:,...]*r[-1:,...]).sum(dim=dim)/(x_norm[1:,...]*x_norm[:-1,...])
        x_norm = (r.pow(2).sum(dim=dim)+1e-5)**0.5
        cs = (r*r[-1]).sum(dim=dim)/(x_norm*x_norm[-1,...])
        cs = cs.mean(0,keepdim=True)
        cs = 1/(cs+1e-5)
        # cs = torch.cat((torch.zeros_like(cs[0:1]), cs),dim=0)

        return  cs

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



