import torch
from torch import nn
import numpy as np


class RCSANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_loss = True
        
    def forward(self,x):
        x = x.unsqueeze(0)
        rank = len(x.size())-2  # N,T,C,W,H
        dim = -np.arange(rank)-1
        dim = list(dim)
        x_clone = torch.maximum(x,torch.tensor(0.0))
        xmax = torch.max(x_clone)
        sigma = (x_clone.pow(2).mean(dim=dim)+1e-5)**0.5
        r = xmax/sigma
        return r


class RCSANNLoss(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_reg_loss(self, x, y, features):
        _target = torch.unsqueeze(y,dim=0) 
        right_predict_mask = x.max(-1)[1].eq(_target).to(torch.float32)
        right_predict_mask = right_predict_mask.prod(0,keepdim=True)
        right_predict_mask = torch.unsqueeze(right_predict_mask,dim=2).flatten(0, 1).contiguous().detach()
        features = features*right_predict_mask
        features = features.max(dim=0)[0]
        loss = features.mean()
        return loss 


