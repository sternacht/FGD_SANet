import torch
import torch.nn as nn
import torch.nn.functional as F

from net.layer import *
from net.SCconv import SCConv3D
from net.eca_module import ECABlock

bn_momentum = 0.1

def Conv_3D(in_plane, out_plane, kernel=3, stride=1, padding=1):
    conv = nn.Sequential(nn.Conv3d(in_plane, out_plane, kernel_size=kernel, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm3d(out_plane, momentum=bn_momentum),
                         nn.ReLU(inplace = True))
    return conv

def DeConv_3D(in_plane, out_plane, kernel_size=2, stride=2):
    # deconv = nn.Sequential(nn.ConvTranspose3d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, bias=False),
    #                      nn.BatchNorm3d(out_plane, momentum=bn_momentum),
    #                      nn.ReLU(inplace = True))
    deconv = nn.Upsample(scale_factor=2, mode='trilinear')
    return deconv

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1, bias=False)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride, bias=False),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out
    
class ECA_SC(nn.Module):
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride = 1,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm3d):
        # parameter for resnet
        super(ECA_SC, self).__init__()

        if stride != 1 or planes != inplanes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size = 1, stride = stride, bias=False),
                nn.BatchNorm3d(planes, momentum=bn_momentum))
        else:
            self.shortcut = None

        # SCconv3D won't change the channel of input here
        self.SCconv1 = SCConv3D(inplanes, planes, stride = stride,
                 cardinality=cardinality, bottleneck_width=bottleneck_width,
                 avd=avd, dilation=dilation, is_first=is_first,
                 norm_layer=norm_layer)
        self.eca1 = ECABlock(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.SCconv2 = SCConv3D(planes, planes, stride = stride,
                 cardinality=cardinality, bottleneck_width=bottleneck_width,
                 avd=avd, dilation=dilation, is_first=is_first,
                 norm_layer=norm_layer)
        self.eca2 = ECABlock(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.SCconv1(x)
        out = self.eca1(out)
        out = self.relu1(out)

        out = self.SCconv2(out)
        out = self.eca2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu2(out)
        return out
    
class PreBlock(nn.Module):
    def __init__(self, in_channels):
        super(PreBlock, self).__init__()
        self.conv1 = Conv_3D(in_channels, 8, kernel=1, padding='same')

        self.conv2_1 = Conv_3D(in_channels, 8, kernel=1, padding='same')
        self.conv2_2 = Conv_3D(8, 8, kernel=3, padding='same')

        self.conv3_1 = Conv_3D(in_channels, 8, kernel=1, padding='same')
        self.conv3_2 = Conv_3D(8, 8, kernel=5, padding='same')

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2_1(x)
        out2 = self.conv2_2(out2)
        
        out3 = self.conv3_1(x)
        out3 = self.conv3_2(out3)

        # dim is important!!!
        out = torch.cat((out1, out2, out3), dim=1)
        return out

class MultiScalAggregation3fs(nn.Module):
    def __init__(self, channels):
        super(MultiScalAggregation3fs, self).__init__()
        min_c = channels[0]

        
        # first block, resizing number of channel to same
        self.channelresize1 = Conv_3D(channels[0], min_c, kernel=3, padding='same')
        self.channelresize2 = Conv_3D(channels[1], min_c, kernel=3, padding='same')
        self.channelresize3 = Conv_3D(channels[2], min_c, kernel=3, padding='same')

        # second block, first multi-sacle aggregation, 3to3
        self.upsample2_1 = DeConv_3D(min_c, min_c, kernel_size=2, stride=2)
        self.upsample2_2 = DeConv_3D(min_c, min_c, kernel_size=2, stride=2)
        self.downsample2_1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.downsample2_2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv2_1 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_2 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_3 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_4 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_5 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_6 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_7 = Conv_3D(min_c, min_c, kernel=3, padding='same')

        # third block, second multi-scale aggregation, 3to1
        self.upsample3 = DeConv_3D(min_c, min_c, kernel_size=2, stride=2)
        self.downsample3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv3_1 = Conv_3D(min_c, min_c, kernel=3,padding='same')
        self.conv3_2 = Conv_3D(min_c, min_c, kernel=3,padding='same')
        self.conv3_3 = Conv_3D(min_c, min_c, kernel=3,padding='same')

        # finally resize to original channel and residual feature.
        self.conv4 = Conv_3D(min_c, channels[1], kernel=3,padding='same')

    def forward(self, fs):
        # first block, resize small, medium and large feature map to same channels
        fs_l = self.channelresize1(fs[0])
        fs_m = self.channelresize2(fs[1])
        fs_s = self.channelresize3(fs[2])

        # second block, multi-scale aggregation
        fs_s2m = self.conv2_1(self.upsample2_1(fs_s))
        fs_s =   self.conv2_2(fs_s)
        fs_m2s = self.conv2_3(self.downsample2_1(fs_m))
        fs_m2l = self.conv2_4(self.upsample2_2(fs_m))
        fs_m =   self.conv2_5(fs_m)
        fs_l2m = self.conv2_6(self.downsample2_2(fs_l))
        fs_l =   self.conv2_7(fs_l)

        fs_s = fs_s + fs_m2s
        fs_m = fs_m + fs_s2m + fs_l2m
        fs_l = fs_l + fs_m2l

        # third block, resize 3 fs to same size and add
        fs_s = self.conv3_1(self.upsample3(fs_s))
        fs_m = self.conv3_2(fs_m)
        fs_l = self.conv3_3(self.downsample3(fs_l))

        out = fs_s + fs_m + fs_l

        # finally
        out = self.conv4(out)
        out = out + fs[1]

        return out
    
class MultiScalAggregation2fs(nn.Module):
    def __init__(self, channels):
        super(MultiScalAggregation2fs, self).__init__()
        min_c = channels[0]

        # first block, resizing number of channel to same
        self.channelresize1 = Conv_3D(channels[0], min_c, kernel=3,padding='same')
        self.channelresize2 = Conv_3D(channels[1], min_c, kernel=3,padding='same')

        # second block, first multi-sacle aggregation, 3to3
        self.upsample2 = DeConv_3D(min_c, min_c, kernel_size=2, stride=2)
        self.downsample2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv2_1 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_2 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_3 = Conv_3D(min_c, min_c, kernel=3, padding='same')
        self.conv2_4 = Conv_3D(min_c, min_c, kernel=3, padding='same')

        # third block, second multi-scale aggregation, 3to1
        self.upsample3 = DeConv_3D(min_c, min_c, kernel_size=2, stride=2)
        self.downsample3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv3_1 = Conv_3D(min_c, min_c, kernel=3,padding='same')
        self.conv3_2 = Conv_3D(min_c, min_c, kernel=3,padding='same')

        # finally resize to original channel and residual feature.
        self.conv4 = Conv_3D(min_c, channels[1], kernel=3,padding='same')

    def forward(self, fs):
        # first block, resize medium and large feature map to same channels
        fs_l = self.channelresize1(fs[0])
        fs_m = self.channelresize2(fs[1])
        # second block, multi-scale aggregation
        fs_m2l = self.conv2_1(self.upsample2(fs_m))
        fs_m =   self.conv2_2(fs_m)
        fs_l2m = self.conv2_3(self.downsample2(fs_l))
        fs_l =   self.conv2_4(fs_l)
        fs_m = fs_m + fs_l2m
        fs_l = fs_l + fs_m2l

        # third block, resize 3 fs to same size and add
        fs_m = self.conv3_1(fs_m)
        fs_l = self.conv3_2(self.downsample3(fs_l))

        out = fs_m + fs_l

        # finally
        out = self.conv4(out)
        out = out + fs[1]
        
        return out
