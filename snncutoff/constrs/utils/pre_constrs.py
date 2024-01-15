from torch import nn
from typing import Type

class PreConstrs(nn.Module):
    def __init__(self, T: int = 4, 
                 module: Type[nn.Module] = None, 
                 multistep: bool = True):
        super().__init__()
        self.T = T
        self.module = module
        self.multistep = multistep
       
    def reshape(self,x):
        if self.multistep:
            new_dim = [int(x.shape[1]*self.T),]
            new_dim.extend(x.shape[2:])
            return x.reshape(new_dim)
        else: 
            return x[0]
        
    def forward(self, x):
        x = self.reshape(x)
        if self.module is not None:
            x = self.module(x)
        return x 