import random
from easycutoff.layers import *

class Conv2dLayer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding,droprate=0.0,bias=True,batch_norm=True):
        super(Conv2dLayer, self).__init__()
        if batch_norm:
            self.fwd = nn.Sequential(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.BatchNorm2d(out_plane),
                nn.Dropout(p=droprate)
            )
        else:
            self.fwd = nn.Sequential(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding,bias=bias),
                nn.Dropout(p=droprate)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.relu(x)
        return x

class LinearLayer(nn.Module):
    def __init__(self,in_plane,out_plane,droprate=0.0,bias=True,batch_norm=True):
        super(LinearLayer, self).__init__()
        if batch_norm:
            self.fwd = nn.Sequential(
                nn.Linear(in_plane,out_plane),
                nn.BatchNorm1d(out_plane),
                nn.Dropout(p=droprate)
            )
        else:
            self.fwd = nn.Sequential(
                nn.Linear(in_plane,out_plane),
                nn.Dropout(p=droprate)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.fwd(x)
        x = self.relu(x)
        return x



class VGGANN(nn.Module):
    def __init__(self, num_classes):
        super(VGGANN, self).__init__()
        #pool = APLayer(2)
        self.features = nn.Sequential(
            # NormLayer(),
            Conv2dLayer(2,64,8,4,3),
            Conv2dLayer(64,64,3,1,1),
            Conv2dLayer(64,128,3,1,1),
            nn.AvgPool2d(2),
            Conv2dLayer(128,256,3,1,1),
            Conv2dLayer(256,256,3,1,1),
            nn.AvgPool2d(2),
            Conv2dLayer(256,512,3,1,1),
            Conv2dLayer(512,512,3,1,1),
            nn.AvgPool2d(2),
            Conv2dLayer(512,512,3,1,1),
            Conv2dLayer(512,512,3,1,1),
            nn.AvgPool2d(2),
            nn.Flatten(1,-1)
        )
        W = int(32/2/2/2/2)
        self.fc = LinearLayer(512*W*W,512)#OutputLayerCurrent
        self.classifier = nn.Linear(512,num_classes)#OutputLayerCurrent
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.fc(x)
        x = self.classifier(x)
        return x

class VGGANN_NCaltech101(nn.Module):
    def __init__(self, output_dim):
        super(VGGANN_NCaltech101, self).__init__()
        pool = nn.AvgPool2d(2)
        self.features = nn.Sequential(
            nn.Conv2d(2,64,8,4,1),
            nn.Conv2d(64,64,3,1,1),
            nn.Conv2d(64,128,3,1,1),
            pool,
            nn.Conv2d(128,256,3,1,1),
            nn.Conv2d(256,256,3,1,1),
            pool,
            nn.Conv2d(256,512,3,1,1),
            nn.Conv2d(512,512,3,1,1),
            pool,
            nn.Conv2d(512,512,3,1,1),
            nn.Conv2d(512,512,3,1,1),
            pool,
        )
        output_dim = 101 
        self.fc = nn.Linear(512*2*3,output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x