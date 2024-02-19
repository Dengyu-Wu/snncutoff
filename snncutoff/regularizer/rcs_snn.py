import torch
from torch import nn
import numpy as np


class RCSSNN(nn.Module):
    def __init__(self,beta=0.3):
        super().__init__()
        self.beta = beta
        self.add_loss = True

    def forward(self,x,mem):
        rank = len(x.size())-2  # T,N,C,W,H
        dim = -np.arange(rank)-1
        dim = list(dim)
        r_t = x+torch.relu(mem)
        index = -int(r_t.shape[0]*self.beta)
        r_d = r_t[index:,...].mean(0,keepdim=True).detach()
        r_d_norm = (r_d.pow(2).sum(dim=dim)+1e-5)**0.5
        r_t_norm = (r_t.pow(2).sum(dim=dim)+1e-5)**0.5
        cs = (r_t*r_d).sum(dim=dim)/(r_t_norm*r_d_norm)
        cs = cs.mean(0,keepdim=True)
        cs = 1/(cs+1e-5)
        return  cs



class RCSSNNLoss(object):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.rcs_n = config.rcs_n

    def compute_reg_loss(self, x, y, features):
        _target = torch.unsqueeze(y,dim=0) 
        index = -int(x.shape[0]*self.rcs_n)
        right_predict_mask = x[index:].max(-1)[1].eq(_target).to(torch.float32)
        right_predict_mask = right_predict_mask.prod(0,keepdim=True)
        right_predict_mask = torch.unsqueeze(right_predict_mask,dim=2).flatten(0, 1).contiguous().detach()
        features = features*right_predict_mask
        features = features.max(dim=0)[0]
        loss = features.mean()
        return loss 
