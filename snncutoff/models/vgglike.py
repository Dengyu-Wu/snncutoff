import torch
import torch.nn as nn

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



class VGG_Gesture(nn.Module):
    def __init__(self,output_dim = 11):
        super(VGG_Gesture, self).__init__()
        pool = nn.MaxPool2d(2)
        self.features = nn.Sequential(
            Conv2dLayer(2,64,4,4,padding='valid'),
            Conv2dLayer(64,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            Conv2dLayer(128,128,3,1,1),
            pool,
            nn.Flatten(1,-1)
        )
        W = int(128/4/2/2/2/2/2)
        self.fc =  LinearLayer(128*W*W,512,droprate=0.0)
        self.classifier = nn.Linear(512,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.fc(x)
        x = self.classifier(x)
        return x


class VGGANN(nn.Module):
    def __init__(self, num_classes):
        super(VGGANN, self).__init__()
        self.features = nn.Sequential(
            Conv2dLayer(2,64,4,4,padding='valid'),
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
        W = int(128/4/2/2/2/2)
        self.classifier = nn.Linear(512*W*W,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.classifier(x)
        return x


class VGGANN_NCaltech101(nn.Module):
    def __init__(self, output_dim=101):
        super(VGGANN_NCaltech101, self).__init__()
        pool = nn.AvgPool2d(2)
        self.features = nn.Sequential(
            Conv2dLayer(2,64,4,4,padding='valid'),
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
        self.fc = nn.Linear(512*2*3,output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = self.fc(x)
        return x