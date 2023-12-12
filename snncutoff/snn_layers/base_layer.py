import torch
from torch import nn
from snncutoff.regularizer import SNNROE
from typing import Type
from snncutoff.gradients import ZIF
from snncutoff.neuron import LIF


class BaseLayer(nn.Module):
    def __init__(self, 
                 T: int = 4, 
                 L: int = 4, 
                 vthr: float = 1.0, 
                 tau: float = 0.5, 
                 neuron: Type[LIF]=LIF,
                 regularizer: Type[SNNROE] = None, 
                 surogate: Type[ZIF] = ZIF,
                 multistep: bool=True
                 ):
        super(BaseLayer, self).__init__()
        
        self.vthr = vthr
        self.T = T
        self.neuron=neuron(vthr=vthr, tau=tau)
        self.reset()
        self.regularizer = regularizer
        self.multistep = multistep
        self.surogate=ZIF.apply
        self.gamma = 1.0

    def mem_init(self,x):
        pass

    def _mem_update_multistep(self,x):
        spike_post = []
        mem_post = []
        self.neuron.reset()
        for t in range(self.T):
            vmem = self.neuron.vmem + x[t]
            spike =  self.surogate(vmem - self.vthr, self.gamma)
            vmem = vmem * (1-spike)
            self.neuron.updateMem(vmem)
            spike_post.append(spike)
            mem_post.append(vmem)
        return torch.stack(spike_post,dim=0), torch.stack(mem_post,dim=0)

    def _mem_update_singlestep(self,x):
            # self.mem_init(x)
            mem = self.neuron.vmem + x 
            x =  (mem > self.vthr).float()
            mem = mem * (1-x)
            self.neuron.updateMem(mem)
            return x, 0.0

    def mem_update(self,x):
        if self.multistep:
            return self._mem_update_multistep(x)
        else:
            return self._mem_update_singlestep(x)   
        
    def forward(self, x):    
        x = self.reshape(x)
        spike_post, mem_post = self.mem_update(x)
        if self.regularizer is not None:
            loss = self.regularizer(spike_post.clone()*self.vthr, mem_post.clone())
        return spike_post 
    
    def reshape(self,x):
        batch_size = int(x.shape[0]/self.T)
        new_dim = [self.T, batch_size]
        new_dim.extend(x.shape[1:])
        return x.reshape(new_dim)

    def reset(self):
        self.neuron.reset()

