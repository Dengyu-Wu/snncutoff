import torch
from torch import nn


class BaseNeuron(nn.Module):
    def __init__(self, vthr=1., tau=1.0):
        super(BaseNeuron, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=tau)
        self.reset()

    def mem_init(self,x):
        pass

    def mem_update(self,x):
        self.mem_init(x)
        mem = self.neuron.mem + x 
        x =  (mem > self.vthr).float()
        mem = mem * (1-x)
        self.neuron.updateMem(mem)
        return x 

    def forward(self, x):    
        x = self.mem_update(x)
        return x 
    
    def reset(self):
        self.neuron.reset()



class IFNeuronReg(IFNeuron):
    def __init__(self, vthr=1.):
        super(IFNeuronReg, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=1.0)
        self.neuron.reset()
    def forward(self, x):    
        x = self.mem_update(x)
        return x, 0.0
    
class LIFNeuron(BaseNeuron):
    def __init__(self, vthr=1.,tau=0.5):
        super(LIFNeuron, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=tau)
        self.reset()

class LIFNeuronReg(BaseNeuron):
    def __init__(self, vthr=1.,tau=0.5):
        super(LIFNeuronReg, self).__init__()
        self.vthr = vthr
        self.neuron=LIF(tau=tau)
        self.neuron.reset()

    def forward(self, x):    
        x = self.mem_update(x)
        return x, 0.0


class LIF(object):
    def __init__(self, tau=0.5):
        self.t = 0.0
        self.mem = 0.0
        self.tau = tau

    def reset(self):
        self.t = 0
        self.mem = 0.0

    def initMem(self,x):
        self.mem = x

    def updateMem(self,x):
        self.mem = x*self.tau
        self.t += 1 
