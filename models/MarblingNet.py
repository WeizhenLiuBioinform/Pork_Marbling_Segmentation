#分块
import torch
import torch.nn as nn
from torchvision.models import vgg13_bn, vgg16_bn
from itertools import chain
from functools import partial
import torch.nn.functional as F
from models.da_att import PAM_Module,CAM_Module

__all__ = ['vgg13bn_unet', 'vgg16bn_unet']

nonlinearity = partial(F.relu, inplace=True)

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class MarblingNet(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """
    def __init__(self, num_classes, in_channels=3, pretrained=True, backbone="vgg16", **_):
        super().__init__()
        if backbone == 'vgg16':
            encoder = vgg16_bn
        elif backbone == 'vgg13':
            encoder = vgg13_bn
        else:
            raise ValueError("Unsupported backbone!")
        self.encoder = encoder(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])#[:6]
        self.block2 = nn.Sequential(*self.encoder[6:13])#[6:13]
        self.block3 = nn.Sequential(*self.encoder[13:23])#[13:23]
        self.block4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                    nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))


        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.up_conv6 = up_conv(516, 256)
        self.conv6 = double_conv(256 + 256, 256)
        self.up_conv7 = up_conv(256, 128)
        self.conv7 = double_conv(128 + 128, 128)
        self.up_conv8 = up_conv(128, 64)
        self.conv8 = double_conv(64 + 64, 64)
        self.conv9 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        

        
    def forward(self, x):
        # outputs = []
        x = F.upsample(x, size=(2*x.size(2),2*x.size(3)), mode='bilinear')
        block1 = self.block1(x)#64,800   
        block2 = self.block2(block1)#128,400
        block3 = self.block3(block2)#256,200
        block4 = self.block4(block3)#512,100
       
        #多尺度特征提取模块
        block4 = self.dblock(block4)#512,100
        block4 = self.spp(block4)#516,100

        x = self.up_conv6(block4)#256,200
        x = torch.cat([x, block3], dim=1)#512,200
        x = self.conv6(x)#256,200

        x = self.up_conv7(x)#128,400
        x = torch.cat([x, block2], dim=1)#256,400
        x = self.conv7(x)#128,400

        x = self.up_conv8(x)#64,800
        x = torch.cat([x, block1], dim=1)#128,800
        x = self.conv8(x)#64,800

        x = self.conv9(x)#2,800*800
        x = self.maxpool(x)

        return x

         
    
    def get_backbone_params(self):
        return chain(
            self.block1.parameters(), self.block2.parameters(), self.block3.parameters(), self.block4.parameters(), self.block5.parameters(),
            self.bottleneck.parameters(), self.conv_bottleneck.parameters()
        )

    def get_decoder_params(self):
        return chain(
            self.up_conv6.parameters(), self.up_conv7.parameters(), self.up_conv8.parameters(), self.up_conv9.parameters(), self.up_conv10.parameters(),
            self.conv6.parameters(), self.conv7.parameters(), self.conv8.parameters(), self.conv9.parameters(), self.conv10.parameters(), self.conv11.parameters()
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()