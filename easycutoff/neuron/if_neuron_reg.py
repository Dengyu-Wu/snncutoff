import torch
from torch import nn
from .base_neuron import *
from .if_neuron import IFNeuron

class IFNeuronReg(IFNeuron):
    def __init__(self, vthr=1.):
        super(IFNeuronReg, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=1.0)
        self.neuron.reset()
    def forward(self, x):    
        x = self.mem_update(x)
        return x, 0.0
    

