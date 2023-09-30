import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


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

class Cov2dLIF(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding,droprate=0.0):
        super(Cov2dLIF, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane),
            nn.Dropout(p=droprate)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)[0]
        return x

class LinearLIF(nn.Module):
    def __init__(self,in_plane,out_plane,droprate=0.0, BatchNorm=True):
        super(LinearLIF, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Linear(in_plane,out_plane),
            nn.BatchNorm1d(out_plane),
            nn.Dropout(p=droprate)
        ) if BatchNorm else SeqToANNContainer(
            nn.Linear(in_plane,out_plane)
        )

        self.act = LIF()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class RecurrentLIF(nn.Module):
    def __init__(self,in_plane,out_plane,droprate=0.0, BatchNorm=True):
        super(RecurrentLIF, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane),
            nn.BatchNorm2d(out_plane),
            nn.Dropout(p=droprate)
        ) if BatchNorm else SeqToANNContainer(
            nn.Linear(in_plane,out_plane)
        )

        self.act = RLIF(out_plane)

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class surrogate(torch.autograd.Function):
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
        alpha = 2.0
        #tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        # tmp = alpha*(1-torch.sigmoid(alpha*input))*torch.sigmoid(alpha*input) # alpha =5
        # tmp = alpha/(2*(1+(torch.pi*alpha*input/2)**2))
        # tmp = torch.exp(-2*input.abs())
        # tmp = 3/(2*(1+3*input.abs()))**2
        tmp = alpha/(2*(1+(torch.pi*alpha*input/2)**2)) # alpha = 2
        grad_input = grad_input * tmp
        return grad_input, None

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

        loss = (spike_pot.clone().pow(2).sum(dim=dim)+1e-7)**0.5/(_pre_mem.clone().pow(2).sum(dim=dim)+1e-7)**0.5
        return spike_pot, loss

class LIF(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIF, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama


    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem =  mem*0.5 + x[:, t]
            spike = self.act(mem - self.thresh, self.gama)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        outputs = torch.stack(spike_pot, dim=1)
        return outputs


class OutputLayerSpike(nn.Module):
    def __init__(self,in_plane, out_plane):
        super(OutputLayerSpike, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Linear(in_plane,out_plane)
        )
        self.act = LIF()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x) 
        return x

class RLIF(nn.Module):
    def __init__(self,out_plane, thresh=1, tau=0.5, gama=1.0):
        super(RLIF, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        weights = torch.empty(out_plane,out_plane,device='cuda')
        self.weights = nn.init.orthogonal_(weights)

    def forward(self, x):
        mem = 0
        spike_pot = []
        spike = torch.zeros_like(x[:,0, ...])
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]+torch.matmul(spike, self.weights)
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x


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


# class LIFSpike(nn.Module):
#     def __init__(self, thresh=1, tau=0.5, gama=1.0):
#         super(LIFSpike, self).__init__()
#         self.act = ZIF.apply

#         self.thresh = thresh
#         self.tau = tau
#         self.gama = gama

#     def forward(self, x):
#         mem = 0
#         spike_pot = []
#         T = x.shape[1]
#         for t in range(T):
#             mem = mem * self.tau + x[:, t, ...]
#             spike = self.act(mem - self.thresh, self.gama)
#             mem = (1 - spike) * mem
#             spike_pot.append(spike)
#         return torch.stack(spike_pot, dim=1)


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


# LIFSpike = LIF
