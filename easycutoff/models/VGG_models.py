import random
from easycutoff.layers import *

class DVSVGG_Current(nn.Module):
    def __init__(self):
        super(DVSVGG_Current, self).__init__()
        pool = SeqToANNContainer(nn.MaxPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Cov2dLIF(2,128,8,4,3),
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool
        )
        W = int(32/2/2/2/2/2)
        self.T = 4
        output_dim = 10 + 1+1
        self.fc1 = SeqToANNContainer(nn.Linear(128*W*W,512), nn.Dropout(p=0.2))
        self.fc2 = SeqToANNContainer(nn.Linear(512,128), nn.Dropout(p=0.2))
        #self.classifier = SeqToANNContainer(nn.Linear(128,output_dim))
        self.classifier  = OutputLayerCurrent(128,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x, confidence = self.classifier(x)
        return x, confidence

class DVSVGG_Spike(nn.Module):
    def __init__(self):
        super(DVSVGG_Spike, self).__init__()
        pool = SeqToANNContainer(nn.MaxPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Cov2dLIF(2,128,8,4,3),
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool,
            Cov2dLIF(128,128,3,1,1),
            pool
        )
        W = int(32/2/2/2/2/2)
        self.T = 4
        output_dim = 10 + 1+1
        self.fc1 = SeqToANNContainer(nn.Linear(128*W*W,512), nn.Dropout(p=0.2))
        self.fc2 = SeqToANNContainer(nn.Linear(512,128), nn.Dropout(p=0.2))
        #self.classifier = SeqToANNContainer(nn.Linear(128,output_dim))
        self.classifier  = OutputLayerSpike(128,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x, confidence = self.classifier(x)
        return x, confidence

class DVSVGG2(nn.Module):
    def __init__(self):
        super(DVSVGG2, self).__init__()
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
        self.T = 4
        output_dim = 10 + 1+1
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,output_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

class VGGSNN_Spike(nn.Module):
    def __init__(self):
        super(VGGSNN_Spike, self).__init__()
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
        self.T = 4
        output_dim = 10 
        output_cutoff = 2
        self.classifier_1 = SeqToANNContainer(nn.Linear(512*W*W,output_dim))#OutputLayerCurrent
        #self.classifier_cutoff  = OutputLayerCurrent(512*W*W,output_dim) #if current_out else OutputLayerSpike(512*W*W,output_dim) 
        # self.classifier  = OutputLayerCurrent(512*W*W,output_dim) #if current_out else OutputLayerSpike(512*W*W,output_dim) 
        self.classifier_2 = SeqToANNContainer(nn.Linear(512*W*W,output_cutoff))#OutputLayerCurrent

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        #input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        confidence = self.classifier_2(x)
        x = self.classifier_1(x)
        return x, confidence

class VGGSNN(nn.Module):
    def __init__(self, current_out=False):
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
        self.T = 4
        output_dim = 10 
        output_cutoff = 1
        self.fc = SeqToANNContainer(nn.Linear(512*W*W,10))#OutputLayerCurrent
        # self.classifier_1  = OutputLayerCurrent(512*W*W,output_dim) #if current_out else OutputLayerSpike(512*W*W,output_dim)         
        #self.classifier_2  = OutputLayerCurrent(512*W*W,1)#if current_out else OutputLayerSpike(512*W*W,output_dim) 

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



if __name__ == '__main__':
    model = VGGSNNwoAP()
    