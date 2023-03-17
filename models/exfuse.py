"""
Exfuse 模块：https://blog.csdn.net/weixin_44543648/article/details/122992755
"""

from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights, set_trainable
from itertools import chain

import torch
from torch import nn


class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0),bias = False)
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1),bias = False)
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x

class SEB(nn.Module):
    """
    语义嵌入模块
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SEB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear")

    def forward(self, low_level, high_level):
        high_level = self.conv(high_level)
        high_level = self.upsample(high_level)
        x = low_level * high_level
        return x


class ResDAP(nn.Module):
    """
    带残差连接密集邻接预测模块
    """
    def __init__(self, in_channels, out_channels, dap_k=2):
        self.DAP = nn.Sequential(
            nn.PixelShuffle(dap_k),
            nn.AvgPool2d((dap_k,dap_k))
        )

class GCNFuse(nn.Module):
    def __init__(self, backbone, kernel_size=7, dap_k=3):
        super(GCNFuse, self).__init__()
        self.num_classes =20
        num_classes = self.num_classes
        self.resnet_features = backbone
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.resnet_features.bn1,
                                    self.resnet_features.relu1, self.resnet_features.conv3,
                                    self.resnet_features.bn3, self.resnet_features.relu3
                                    )
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes * 4, (kernel_size, kernel_size))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (kernel_size, kernel_size))
        self.gcm3 = _GlobalConvModule(512, num_classes * dap_k**2, (kernel_size, kernel_size))
        self.gcm4 = _GlobalConvModule(256, num_classes * dap_k**2, (kernel_size, kernel_size))

        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(num_classes * dap_k**2, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(num_classes * dap_k**2, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(num_classes * dap_k**2, num_classes * dap_k**2, kernel_size=4, stride=2, padding=1, bias=False)

        self.ecre = nn.PixelShuffle(2)

        self.seb1 = SEB(2048, 1024)
        self.seb2 = SEB(3072, 512)
        self.seb3 = SEB(3584, 256)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.DAP = nn.Sequential(
            nn.PixelShuffle(dap_k),
            nn.AvgPool2d((dap_k,dap_k))
        )

    def forward(self, x):
        # suppose input = x , if x 512
        f0 = self.layer0(x)  # 256
        f1 = self.layer1(f0)  # 128
        print (f1.size())
        f2 = self.layer2(f1)  # 64
        print (f2.size())
        f3 = self.layer3(f2)  # 32
        print (f3.size())
        f4 = self.layer4(f3)  # 16
        print (f4.size())
        x = self.gcm1(f4)
        out1 = self.ecre(x)
        seb1 = self.seb1([f3, f4])
        gcn1 = self.gcm2(seb1)

        seb2 = self.seb2([f2, torch.cat([f3, self.upsample2(f4)], dim=1)])
        gcn2 = self.gcm3(seb2)

        seb3 = self.seb3([f1, torch.cat([f2, self.upsample2(f3), self.upsample4(f4)], dim=1)])
        gcn3 = self.gcm4(seb3)

        y = self.deconv2(gcn1 + out1)
        y = self.deconv3(gcn2 + y)
        y = self.deconv4(gcn3 + y)
        y = self.deconv5(y)
        y = self.DAP(y)
        return y

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == '__main__':
    model = GCNFuse(20)
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True)
    res1= model(image)
    print (res1.size())

