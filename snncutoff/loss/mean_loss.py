
import torch
import torch.nn as nn


class MeanLoss(nn.Module):
    def __init__(self, criterion, *args, **kwargs):
        super(MeanLoss, self).__init__()
        self.criterion=criterion

    def forward(self, x, y):
        mean = x.mean(0)
        loss = self.criterion(mean,y)
        return mean, loss