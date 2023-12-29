import random
from snncutoff.layers import *

class VGGSNN(nn.Module):
    def __init__(self, output_dim):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Cov2dLIF(2,64,8,4,3),
            Cov2dLIF(64,64,3,1,1),
            Cov2dLIF(64,128,3,1,1),
            pool,
            Cov2dLIF(128,256,3,1,1),
            Cov2dLIF(256,256,3,1,1),
            pool,
            Cov2dLIF(256,512,3,1,1),
            Cov2dLIF(512,512,3,1,1),
            pool,
            Cov2dLIF(512,512,3,1,1),
            Cov2dLIF(512,512,3,1,1),
            pool,
        )
        W = int(32/2/2/2/2)
        self.fc = SeqToANNContainer(nn.Linear(512*W*W,output_dim))#OutputLayerCurrent
        # self.classifier_1  = OutputLayerCurrent(512*W*W,output_dim) #if current_out else OutputLayerSpike(512*W*W,output_dim)         
        #self.classifier_2  = OutputLayerCurrent(512*W*W,1)#if current_out else OutputLayerSpike(512*W*W,output_dim) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x




class DVSVGG(nn.Module):
    def __init__(self):
        super(DVSVGG, self).__init__()
        pool = SeqToANNContainer(nn.MaxPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2,128,8,4,3),
            Layer(128,128,3,1,1),
            pool,
            Layer(128,128,3,1,1),
            pool,
            Layer(128,128,3,1,1),
            pool,
            Layer(128,128,3,1,1),
            pool,
            Layer(128,128,3,1,1),
            pool
        )
        W = int(32/2/2/2/2/2)
        output_dim = 10 + 1
        self.fc1 = SeqToANNContainer(nn.Linear(128*W*W,512),nn.Dropout(0.2))
        self.fc2 = SeqToANNContainer(nn.Linear(512,output_dim))
        self.spike=LIFSpike()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        x = self.spike(x)[0]
        x = self.fc2(x)
        return x


class VGGANN_m(nn.Module):
    def __init__(self, num_classes):
        super(VGGANN_m, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            # NormLayer(),
            Cov2dReLU(2,64,8,4,3),
            Cov2dReLU(64,64,3,1,1),
            Cov2dReLU(64,128,3,1,1),
            SeqToANNContainer(nn.AvgPool2d(2)),
            Cov2dReLU(128,256,3,1,1),
            Cov2dReLU(256,256,3,1,1),
            SeqToANNContainer(nn.AvgPool2d(2)),
            Cov2dReLU(256,512,3,1,1),
            Cov2dReLU(512,512,3,1,1),
            SeqToANNContainer(nn.AvgPool2d(2)),
            Cov2dReLU(512,512,3,1,1),
            Cov2dReLU(512,512,3,1,1),
            SeqToANNContainer(nn.AvgPool2d(2)),
            nn.Flatten(2,-1),
        )
        W = int(32/2/2/2/2)
        self.fc1 = LinearReLU(512*W*W,512)#OutputLayerCurrent
        self.fc = SeqToANNContainer(nn.Linear(512,num_classes))#OutputLayerCurrent
        # self.classifier_1  = OutputLayerCurrent(512*W*W,output_dim) #if current_out else OutputLayerSpike(512*W*W,output_dim)         
        #self.classifier_2  = OutputLayerCurrent(512*W*W,1)#if current_out else OutputLayerSpike(512*W*W,output_dim) 
        # self.flatten = nn.Flatten(2,-1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            # if isinstance(m, nn.Linear):
            #     nn.init.kaiming_normal_(m.weight.data)# mode='fan_out', nonlinearity='relu')
    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        # x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc(x)
        return x


class VGG11(nn.Module):
    def __init__(self, current_out=False):
        super(VGG11, self).__init__()
        pool = SeqToANNContainer(nn.MaxPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            NormLayer(),
            Cov2dReLU(2,64,8,4,3),
            Cov2dReLU(64,64,3,1,1),
            pool,
            Cov2dReLU(64,128,3,1,1),
            pool,
            Cov2dReLU(128,256,3,1,1),
            Cov2dReLU(256,256,3,1,1),
            pool,
            Cov2dReLU(256,512,3,1,1),
            Cov2dReLU(512,512,3,1,1),
            pool,
            Cov2dReLU(512,512,3,1,1),
            Cov2dReLU(512,512,3,1,1),
            pool,
        )
        W = int(32/2/2/2/2/2)
        self.T = 4
        output_dim = 10 
        output_cutoff = 1
        self.fc1 = LinearReLU(512*W*W,512)#OutputLayerCurrent
        self.fc = SeqToANNContainer(nn.Linear(512,10))#OutputLayerCurrent
        # self.classifier_1  = OutputLayerCurrent(512*W*W,output_dim) #if current_out else OutputLayerSpike(512*W*W,output_dim)         
        #self.classifier_2  = OutputLayerCurrent(512*W*W,1)#if current_out else OutputLayerSpike(512*W*W,output_dim) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        x = self.fc(x)
        return x




class VGGSNN_NCaltech101(nn.Module):
    def __init__(self, current_out=False):
        super(VGGSNN_NCaltech101, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Cov2dLIF(2,64,8,4,1),
            Cov2dLIF(64,64,3,1,1),
            Cov2dLIF(64,128,3,1,1),
            pool,
            Cov2dLIF(128,256,3,1,1),
            Cov2dLIF(256,256,3,1,1),
            pool,
            Cov2dLIF(256,512,3,1,1),
            Cov2dLIF(512,512,3,1,1),
            pool,
            Cov2dLIF(512,512,3,1,1),
            Cov2dLIF(512,512,3,1,1),
            pool,
        )
        output_dim = 101 
        self.fc = SeqToANNContainer(nn.Linear(512*2*3,output_dim))#OutputLayerCurrent

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x



class VGGANN_NCaltech101(nn.Module):
    def __init__(self, current_out=False):
        super(VGGSNN_NCaltech101, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Cov2dReLU(2,64,8,4,1),
            Cov2dReLU(64,64,3,1,1),
            Cov2dReLU(64,128,3,1,1),
            pool,
            Cov2dReLU(128,256,3,1,1),
            Cov2dReLU(256,256,3,1,1),
            pool,
            Cov2dReLU(256,512,3,1,1),
            Cov2dReLU(512,512,3,1,1),
            pool,
            Cov2dReLU(512,512,3,1,1),
            Cov2dReLU(512,512,3,1,1),
            pool,
        )
        output_dim = 101 
        self.fc = SeqToANNContainer(nn.Linear(512*2*3,output_dim))#OutputLayerCurrent

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

# class VGGSNN_TEBN(nn.Module):
#     def __init__(self, current_out=False):
#         super(VGGSNN_TEBN, self).__init__()
#         pool = SeqToANNContainer(nn.AvgPool2d(2))
#         #pool = APLayer(2)
#         self.features = nn.Sequential(
#             TEBNLayer(2,64,8,4,3),
#             TEBNLayer(64,64,3,1,1),
#             TEBNLayer(64,128,3,1,1),
#             pool,
#             TEBNLayer(128,256,3,1,1),
#             TEBNLayer(256,256,3,1,1),
#             pool,
#             TEBNLayer(256,512,3,1,1),
#             TEBNLayer(512,512,3,1,1),
#             pool,
#             TEBNLayer(512,512,3,1,1),
#             TEBNLayer(512,512,3,1,1),
#             pool,
#         )
#         W = int(32/2/2/2/2)
#         self.T = 4
#         output_dim = 10 
#         output_cutoff = 1
#         self.fc = SeqToANNContainer(nn.Linear(512*W*W,512))#OutputLayerCurrent
#         self.classifier_1 = SeqToANNContainer(nn.Linear(512,output_dim))#OutputLayerCurrent
#         # self.classifier_1  = OutputLayerCurrent(512*W*W,output_dim) #if current_out else OutputLayerSpike(512*W*W,output_dim)         
#         #self.classifier_2  = OutputLayerCurrent(512*W*W,1)#if current_out else OutputLayerSpike(512*W*W,output_dim) 

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, input):
#         #input = add_dimention(input, self.T)
#         x = self.features(input)
#         x = torch.flatten(x, 2)
#         x = self.fc(x)
#         x = self.classifier_1(x)
#         return x


# class VGGSNNwoAP(nn.Module):
#     def __init__(self):
#         super(VGGSNNwoAP, self).__init__()
#         self.features = nn.Sequential(
#             Layer(2,64,3,1,1),
#             Layer(64,128,3,2,1),
#             Layer(128,256,3,1,1),
#             Layer(256,256,3,2,1),
#             Layer(256,512,3,1,1),
#             Layer(512,512,3,2,1),
#             Layer(512,512,3,1,1),
#             Layer(512,512,3,2,1),
#         )
#         W = int(48/2/2/2/2)
#         self.T = 4
#         self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10+1))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, input):
#         #input = add_dimention(input, self.T)
#         x = self.features(input)
#         x = torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x
