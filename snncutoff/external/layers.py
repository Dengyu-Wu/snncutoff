import torch
import torch.nn as nn

class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding,droprate=0.0):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane),
            nn.Dropout(p=droprate)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x    

class LIFSpike(nn.Module):
    def __init__(self, thresh=1, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        pre_mem = 0
        spike_pot = []
        T = x.shape[0]
        _pre_mem = []
        import numpy as np
        rank = len(x.size())-1   # N,T,C,W,H
        rank = np.clip(rank,3,rank)
        dim = np.arange(rank-1)+2   
        dim = list(dim)
        for t in range(T):
            mem = pre_mem * self.tau + x[t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            pre_mem = (1 - spike) * mem
            _pre_mem.append(pre_mem)
            spike_pot.append(spike)
        spike_pot = torch.stack(spike_pot, dim=0)
        _pre_mem = torch.stack(_pre_mem, dim=0)

        return spike_pot
