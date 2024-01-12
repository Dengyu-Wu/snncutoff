import torch
import torch.nn as nn
from snncutoff.external.layers import Layer, SeqToANNContainer

class VGGSNN(nn.Module):
    def __init__(self, current_out=False):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2,64,8,4,3),   # Downscaling layer
            Layer(64,64,3,1,1),
            Layer(64,128,3,1,1),
            pool,
            Layer(128,256,3,1,1),
            Layer(256,256,3,1,1),
            pool,
            Layer(256,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
            Layer(512,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
        )
        W = int(32/2/2/2/2)
        self.T = 4
        self.fc = SeqToANNContainer(nn.Linear(512*W*W,10))#OutputLayerCurrent

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x
