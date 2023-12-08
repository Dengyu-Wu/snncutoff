import torch
from torch import nn
from .base_neuron import *

class IFNeuron(BaseNeuron):
    def __init__(self, vthr=1.):
        super(IFNeuron, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=1.0)
        self.neuron.reset()

    def mem_init(self,x):
        if self.neuron.t == 0:
            self.neuron.initMem(torch.ones_like(x)*0.5)

    def mem_update(self,x):
        x = x / self.vthr
        self.mem_init(x)
        mem = self.neuron.mem + x 
        x =  (mem > 1.0).float()
        mem = mem - 1.0*x
        self.neuron.updateMem(mem)
        x = x * self.vthr
        return x 
    
    def forward(self, x):    
        x = self.mem_update(x)
        return x 
    
    def reset(self):
        self.neuron.reset()

        