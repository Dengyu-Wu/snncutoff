import torch
from torch import nn
from .base_neuron import *

class LIFNeuron(BaseNeuron):
    def __init__(self, vthr=1.,tau=0.5):
        super(LIFNeuron, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=tau)
        self.reset()

        