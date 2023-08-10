from resnet_cbam import resnet50CBAM
from basicblock import basic_block
from resnet_senet import resnet50SE
import torch.nn as nn
import torch
import torch.nn.functional as F

# Mish activation function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#   BasicConv -> Conv2d + BatchNormalization + Mish
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class CrossNet(nn.Module):
    def __init__(self):
        super(CrossNet, self).__init__()
        # initialize the resnet50CBAM used to touch modality feature extraction
        self.modality = resnet50CBAM()
        # initialize the resnet50SENet used to vision modality feature extraction
        self.modality2 = resnet50SE()

        # ConvBlock contains four BasicConvs
        self.ConvBlock = basic_block()

        self.basicconv = BasicConv(512, 512, 1)
        self.basicconv2 = BasicConv(1024, 512, 1)

        self.basicconv3 = BasicConv(512, 512, 1)
        self.basicconv4 = BasicConv(1024, 512, 1)


    def forward(self, x1, x2):
        # touch modality feature extraction
        x1 = self.modality(x1)
        # vision modality feature extraction
        x2 = self.modality2(x2)

        # Cross-attention Mechanism
        x = torch.cat((x1,x2), dim=1)
        x = self.ConvBlock(x)


        x3 = self.basicconv(x)
        # concat feature between output from initial touch modality feature
        # and output after ConvBlock
        x1 = torch.cat((x1, x3), dim=1)
        # output for touch modality
        x1 = self.basicconv2(x1)

        x4 = self.basicconv3(x)
        # concat feature between output from initial vision modality feature
        # and output after ConvBlock
        x2 = torch.cat((x2, x4), dim=1)
        # output for vision modality
        x2 = self.basicconv4(x2)


        return x1, x2