import torch
import torch.nn as nn

#-------------------------------------------------#
# Channel Attention Module
#-------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        # Create AvgPool and MaxPool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Changed shared MLP to shared convolutional layer in Convolutional Block Attention Module(CBAM) paper
        self.conv1   = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2   = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        # Create activation function Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Use average pool and maximum pool on spatial axis
        avg_out = self.avg_pool(x)
        avg_out = self.conv1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.conv2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.conv1(max_out)
        max_out = self.relu(max_out)
        max_out = self.conv2(max_out)
        #  Combine AvgPool and MaxPool
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

#-------------------------------------------------#
# Spatial Attention Module
#-------------------------------------------------#
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1,  kernel_size=7, padding=3, bias=False)
        # Create activation function Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Use average pool and maximum pool on channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Combine AvgPool and MaxPool
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

#-------------------------------------------------#
# Convolutional Block Attention Block
#-------------------------------------------------#
class cbam_block(nn.Module):
    def __init__(self, in_channel, ratio=8):
        super(cbam_block, self).__init__()
        # Create channel attention module and spatial attention module
        self.channelattention = ChannelAttention(in_channel, ratio=ratio)
        self.spatialattention = SpatialAttention()

    def forward(self, x):
        # Apply channel attention to feature map
        x = x * self.channelattention(x)
        # Apply spatial attention to feature map
        x = x * self.spatialattention(x)
        return x

#-------------------------------------------------#
# Squeeze-and-Excitation Block
#-------------------------------------------------#
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        # Create AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


