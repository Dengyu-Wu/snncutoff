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
                 mem_init: float = 0., 
                 neuron: Type[LIF]=LIF,
                 regularizer: Type[SNNROE] = None, 
                 surogate: Type[ZIF] = ZIF,
                 multistep: bool=True,
                 reset_mode: str = 'hard'
                 ):
        super(BaseLayer, self).__init__()
        
        self.vthr = vthr
        self.tau = tau
        self.T = T
        self.mem_init = mem_init 
        self.neuron=neuron(vthr=vthr, tau=tau)
        self.neuron.reset()
        self.regularizer = regularizer
        self.multistep = multistep
        self.surogate=ZIF.apply
        self.gamma = 1.0
        self.reset_mode = reset_mode

    def _mem_update_multistep(self,x):
        spike_post = []
        mem_post = []
        self.neuron.reset()
        for t in range(self.T):
            vmem = self.neuron.vmem + x[t]
            spike =  self.surogate(vmem - self.vthr, self.gamma)
            vmem = self.vmem_reset(vmem,spike)
            self.neuron.updateMem(vmem)
            spike_post.append(spike)
            mem_post.append(vmem)
        return torch.stack(spike_post,dim=0), torch.stack(mem_post,dim=0)
    
    def vmem_reset(self, x, spike):
        if self.reset_mode == 'hard':
            return x * (1-spike)
        elif self.reset_mode == 'soft':  
            return x - self.vthr*spike
        
    def _mem_update_singlestep(self,x):
        if self.neuron.t == 0:
            self.mem_init = 0.5 if self.reset_mode == 'soft' else self.mem_init
            self.neuron.initMem(self.mem_init*self.vthr)
        spike_post = []
        vmem = self.neuron.vmem + x[0]
        spike =  (vmem > self.vthr).float()
        vmem = self.vmem_reset(vmem,spike)
        self.neuron.updateMem(vmem)
        spike_post.append(spike*self.vthr)
        return torch.stack(spike_post,dim=0), 0.0

    def mem_update(self,x):
        if self.multistep:
            return self._mem_update_multistep(x)
        else:
            return self._mem_update_singlestep(x)   
        
    def forward(self, x):  
        x = self.reshape(x)
        spike_post, mem_post = self.mem_update(x)
        if self.regularizer is not None:
            loss = self.regularizer(spike_post.clone(), mem_post.clone()/self.vthr)
        return spike_post
         
    
    def reshape(self,x):
        if self.multistep:
            batch_size = int(x.shape[0]/self.T)
            new_dim = [self.T, batch_size]
            new_dim.extend(x.shape[1:])
            return x.reshape(new_dim)
        else:
            return x.unsqueeze(0)
        


