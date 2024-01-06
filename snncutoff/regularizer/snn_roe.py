import torch
from torch import nn
import numpy as np


class SNNROE(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_loss = True

    def forward(self,x,mem):
        rank = len(x.size())-2  # T,N,C,W,H
        dim = -np.arange(rank)-1
        dim = list(dim)
        # x_norm = (x_clone.pow(2).sum(dim=dim)+1e-5)**0.5
        # cs = (x_clone[1:,...]*x_clone[-1:,...]).sum(dim=dim)/(x_norm[1:,...]*x_norm[:-1,...])
        r_t = x+torch.relu(mem)
        # r_d = r_t.mean(0,keepdim=True).detach()
        r_d = r_t[-1:].detach()

        # r_d = r_t.mean(0,keepdim=True).detach()
        # x_norm = (r.pow(2).sum(dim=dim)+1e-5)**0.5
        # cs = (r[1:,...]*r[-1:,...]).sum(dim=dim)/(x_norm[1:,...]*x_norm[:-1,...])
        r_d_norm = (r_d.pow(2).sum(dim=dim)+1e-5)**0.5
        r_t_norm = (r_t.pow(2).sum(dim=dim)+1e-5)**0.5
        cs = (r_t*r_d).sum(dim=dim)/(r_t_norm*r_d_norm)
        # weight = (torch.arange(cs.size()[0])+1).to(cs.device)
        # cs = cs*(weight.unsqueeze(1))/weight.sum()
        cs = cs.mean(0,keepdim=True)
        # cs = cs.sum(0,keepdim=True)
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



