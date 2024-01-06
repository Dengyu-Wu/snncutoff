import torch.nn as nn
import torch
from .base_layer import BaseLayer

class TEBNLayer(BaseLayer):
    def __init__(self, T, num_features, eps=1e-5, momentum=0.1):
        super(TEBNLayer, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.p = nn.Parameter(torch.ones(T, 1, 1, 1, 1))
        self.T = T
        
    def forward_mult_step(self, input):
        y = self.reshape(input)
        y = y.transpose(0, 1).contiguous()  # T N C H W ,  N T C H W
        y = y.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        y = y.contiguous().transpose(1, 2)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        new_dim = [int(y.shape[1]*self.T),]
        new_dim.extend(y.shape[2:])
        return y.reshape(new_dim)
    
    def forward(self,x):
        if self.multistep:
            return self.forward_mult_step(x)
        else:
            TypeError('Single Step Inference is not supported for TEBN.')
