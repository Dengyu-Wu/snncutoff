
import torch
import torch.nn as nn


class TETLoss(nn.Module):
    def __init__(self, criterion, means, lamb):
        super(TETLoss, self).__init__()
        self.criterion=criterion
        self.means=means
        self.lamb=lamb

    def forward(self, x, y):
        T = x.size(0)
        Loss_es = 0
        for t in range(T):
            Loss_es += self.criterion(x[t, ...], y)
        Loss_es = Loss_es / T # L_TET
        if self.lamb != 0:
            MMDLoss = torch.nn.MSELoss()
            y = torch.zeros_like(x).fill_(self.means)
            Loss_mmd = MMDLoss(x, y) # L_mse
        else:
            Loss_mmd = 0
        return x.mean(0), (1 - self.lamb) * Loss_es + self.lamb * Loss_mmd # L_Total