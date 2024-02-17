"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,multistep=True):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        #shortcut
        self.shortcut = nn.Sequential()
        self.a = nn.Sequential()
        self.shortcut_true = False
        self.multistep = multistep
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut_true = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x 
        out = self.residual_function(x) 
        if self.multistep:
            new_shape = [int(shortcut.size()[1]*shortcut.size()[0]),]
            new_shape.extend(shortcut.size()[2:])
            shortcut = shortcut.view(new_shape)

        shortcut = self.shortcut(x) if self.shortcut_true else shortcut
        out =  out + shortcut
        return self.relu(out)

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, multistep=True):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),

        )
        self.shortcut = nn.Sequential()
        self.multistep = multistep
        self.shortcut_true = False 
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut_true = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x 
        out = self.residual_function(x) 
        if self.multistep:
            new_shape = [int(shortcut.size()[1]*shortcut.size()[0]),]
            new_shape.extend(shortcut.size()[2:])
            shortcut = shortcut.view(new_shape)

        shortcut = self.shortcut(x) if self.shortcut_true else shortcut
        out =  out + shortcut
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_block,input_size, num_classes=100, multistep=True):
        super().__init__()
        self.in_channels = 64
        self.multistep = multistep
        if input_size != 224:
            # This is for input size 3*32*32 and 3*64*64
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
            
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride,self.multistep))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        if self.maxpool is not None:
            output = self.maxpool(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.flatten(output)
        output = self.fc(output)
        return output

class ResNet4Cifar(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def resnet18(num_classes=10, **kargs):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
def resnet20(num_classes=10, **kargs):
    """ return a ResNet 20 object
    """
    return ResNet4Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet34(num_classes=10, **kargs):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10, **kargs):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=10, **kargs):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)

def resnet152(num_classes=10, **kargs):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=num_classes)


cfg = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]

}

def get_resnet(name, input_size=32, num_classes=10, multistep=True, **kargs):
    Block = BasicBlock if name == 'resnet18' or name =='resnet34' else BottleNeck
    return ResNet(Block, cfg[name],input_size=input_size, num_classes=num_classes,multistep=multistep)
