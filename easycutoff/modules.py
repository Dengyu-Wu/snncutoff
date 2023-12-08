from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Function
# from .spikingjelly.clock_driven import neuron

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

class Neuron(object):
    def __init__(self, t=1., s=1):
        self.t = 0.0
        self.mem = 0.0
    
    def reset(self):
        self.t = 0.0
        self.mem = 0.0
        
    def updateT(self):
        self.t += 1 

    def updateMem(self,x):
        self.mem = x

        
class ScaledNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        # self.t = 0.0
        # self.mem = 0.0
        # self.t = 0
        # self.neuron = neuron.IFNode(v_reset=None)
        self.neuron=Neuron()
        self.reset()
    def forward(self, x):    
        
        # print(self.mem)
        x = x / self.scale
        if self.neuron.t == 0:
            # self.updateMem(torch.ones_like(x)*0.5)
            self.neuron.updateMem(torch.ones_like(x)*0.5)
            # self.neuron(torch.ones_like(x)*0.5)
        mem = self.neuron.mem + x #+ torch.ones_like(x)/2.0
        x =  (mem > 1.0).float()
        mem = mem - 1.0*x
        self.neuron.updateMem(mem)
        # x = torch.clamp(x,0,1)
        # x = x * self.scale
        # self.updateMem(mem)
        x = x * self.scale

        self.neuron.updateT()
        # print(self.neuron.t)
        return x # * self.scale
    
    def reset(self):
        self.neuron.reset()



class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class ShiftNeuron(nn.Module):
    def __init__(self, scale=1., alpha=1/50000):
        super().__init__()
        self.alpha = alpha
        self.vt = 0.
        self.scale = scale
        self.neuron = neuron.IFNode(v_reset=None)
    def forward(self, x):  
        x = x / self.scale
        x = self.neuron(x)
        return x * self.scale
    def reset(self):
        if self.training:
            self.vt = self.vt + self.neuron.v.reshape(-1).mean().item()*self.alpha
        self.neuron.reset()
        if self.training == False:
            self.neuron.v = self.vt

class MyFloor(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(x)
        x = x / self.up
        x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

class TCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
    def forward(self, x):
        x = F.relu(x, inplace='True')
        x = self.up - x
        x = F.relu(x, inplace='True')
        x = self.up - x
        return x

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
