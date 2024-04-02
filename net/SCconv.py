import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



class Self_Cailbration(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer,kernel_size=3):
        """
        inplanes: input channels
        planes: output channels
        pooling_r: downsmapling rate before k2
        """
        super(Self_Cailbration, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv3d(inplanes, planes, kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size, stride=1, padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(planes),
        )
    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x),  identity.size()[2:])))  # Sigmoid(identity + k2) with proper upsampling
        out = torch.mul(self.k3(x), out) # k3 * Sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out
    
class SCConv3D(nn.Module):
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.
    def __init__(self, inplanes, planes, stride = 1,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm3d):
        super(SCConv3D, self).__init__()
        # parameter for SCconv
        group_width = int(planes * (bottleneck_width/64.)) * cardinality
        self.conv1_a = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)
        if self.avd:
            self.avd_layer = nn.AvgPool3d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv3d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.sc = Self_Cailbration(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv3d(
            group_width*2, planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dilation = dilation


    def forward(self, x):
        # for scconv

        # self-cailbration branch
        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_a = self.relu(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_b = self.relu(out_b)

        # combine
        out_a = self.k1(out_a)
        out_b = self.sc(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        # do conv3 or not??? should be experimented
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)
        out = self.relu(out)
        return out
    
class Multi_SCConv3D(nn.Module):
    pooling_r = (2,4,8) # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.
    def __init__(self, inplanes, planes, stride = 1,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm3d):
        super(Multi_SCConv3D, self).__init__()
        # parameter for SCconv
        group_width = int(planes * (0.25)) * cardinality
        self.conv1_a = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.conv1_c = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_c = norm_layer(group_width)
        self.conv1_d = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_d = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)
        if self.avd:
            self.avd_layer = nn.AvgPool3d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv3d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.sc2 = Self_Cailbration(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer, kernel_size=3)
        self.sc4 = Self_Cailbration(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer,kernel_size=3)
        self.sc8 = Self_Cailbration(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer,kernel_size=3)
        self.conv3 = nn.Conv3d(
            group_width*4 , planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dilation = dilation


    def forward(self, x):
        # for scconv
        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_c = self.conv1_c(x)
        out_c = self.bn1_c(out_c)
        out_d = self.conv1_d(x)
        out_d = self.bn1_d(out_d)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)
        out_c = self.relu(out_c)
        out_d = self.relu(out_d)

        out_a = self.k1(out_a)
        out_b = self.sc2(out_b)
        out_c = self.sc4(out_c)
        out_d = self.sc8(out_d)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)
        out_c = self.relu(out_c)
        out_d = self.relu(out_d)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)
            out_c = self.avd_layer(out_c)
            out_d = self.avd_layer(out_d)

        # do conv3 or not??? should be experimented
        out = self.conv3(torch.cat([out_a, out_b,out_c,out_d], dim=1))
        out = self.bn3(out)
        out = self.relu(out)
        return out