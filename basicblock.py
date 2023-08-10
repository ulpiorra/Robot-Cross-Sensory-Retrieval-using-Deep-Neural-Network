import torch
import torch.nn as nn
import torch.nn.functional as F

#  Mish activation function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#  BasicConv -> Conv2d + BatchNormalization + Mish
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

# BasicBlock contain four BasicConvs
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()

        self.conv1 = BasicConv(in_channels, out_channels, 3)
        self.conv2 = BasicConv(out_channels, out_channels, 1)
        self.conv3 = BasicConv(out_channels, out_channels, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

def basic_block():
    return BasicBlock(in_channels=1024, out_channels=512)