import torch
from torch import nn
from .base_neuron import *

class LIFNeuronReg(BaseNeuron):
    def __init__(self, vthr=1.,tau=0.5):
        super(LIFNeuronReg, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=tau)
        self.neuron.reset()

    def forward(self, x):    
        x = self.mem_update(x)
        return x, 0.0