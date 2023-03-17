#########################################################################################################
# Copyright (c) 2017 Hang Zhang. All rights reserved.
# Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
# Copyright (c) 2018 Jun Fu. All rights reserved.
# Released under the MIT license
# https://github.com/junfu1115/DANet/blob/master/LICENSE
#
# Reference: Dual Attention Network for Scene Segmentation,
#            Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu, in CVPR2019
#            arXiv: https://arxiv.org/pdf/1809.02983.pdf
#########################################################################################################

# copyright (c)
# https://github.com/yiskw713/DualAttention_for_Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
import torch.nn as nn
from itertools import chain
from .drnet import drn_d_22, drn_d_38, drn_d_54
from .resnet import resnet50, resnet34

from utils.helpers import initialize_weights, set_trainable


class PositionAttentionModule(nn.Module):
    ''' self-attention '''

    def __init__(self, in_channels):
        super().__init__()
        self.chanel_in = in_channels
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """

        N, C, H, W = x.shape
        query = self.query_conv(x).view(
            N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
        key = self.key_conv(x).view(N, -1, H*W)  # (N, C', H*W)

        # caluculate correlation
        energy = torch.bmm(query, key)    # (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H*W)    # (N, C, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.chanel_in = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class DANet(nn.Module):
    def __init__(self, out_channels=4, backbone="resnet50", in_channels=2048, pretrained=False,**_):
        super(DANet, self).__init__()
        inter_channel = in_channels // 4
        # set a base model
        if backbone == 'drn_d_22':
            print('Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True,num_classes=out_channels)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        elif backbone == 'drn_d_38':
            print('Dilated ResNet D 38 wil be used as a base model')
            self.base = drn_d_38(pretrained=True,num_classes=out_channels)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        elif backbone == 'resnet50':
            print("Resnet 50 will be used as a base model")
            self.base = resnet50(pretrained=pretrained, num_classes=out_channels)
            self.base = nn.Sequential(
                *list(self.base.children())[:-2]
            )
        else:
            print('There is no option you choose as a base model.')
            print('Instead, Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True,num_classes=out_channels)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])

        # convolution before attention modules
        self.conv2pam = nn.Sequential(
            nn.Conv2d(in_channels, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )
        self.conv2cam = nn.Sequential(
            nn.Conv2d(in_channels, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )

        # attention modules
        self.pam = PositionAttentionModule(in_channels=inter_channel)
        self.cam = ChannelAttentionModule(in_channels=inter_channel)

        # convolution after attention modules
        self.pam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())
        self.cam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())

        # output layers for each attention module and sum features.
        self.conv_pam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, out_channels, 1)
        )
        self.conv_cam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, out_channels, 1)
        )
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, out_channels, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.base(x)

        # outputs from attention modules
        pam_out = self.conv2pam(x)
        pam_out = self.pam(pam_out)
        pam_out = self.pam2conv(pam_out)
        sa_output = self.conv_pam_out(pam_out)

        cam_out = self.conv2cam(x)
        cam_out = self.cam(cam_out)
        cam_out = self.cam2conv(cam_out)
        sc_output = self.conv_cam_out(cam_out)

        # segmentation result
        feats_sum = pam_out + cam_out
        output = self.conv_out(feats_sum)
        output = F.upsample_bilinear(output, (h, w))
        
        return output
    def get_backbone_params(self):
        return chain(
            self.base.parameters(),
            self.conv2pam.parameters(), self.pam.parameters(), self.pam2conv.parameters(),
            self.conv2cam.parameters(), self.cam.parameters(), self.cam2conv.parameters(),
            self.conv_out.parameters()
        )

    def get_decoder_params(self):
        return []

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
        
       

    # def forward(self, x):
    #     b, c, h, w = x.size()
    #     x = self.base(x)

    #     # outputs from attention modules
    #     pam_out = self.conv2pam(x)
    #     pam_out = self.pam(pam_out)
    #     pam_out = self.pam2conv(pam_out)

    #     cam_out = self.conv2cam(x)
    #     cam_out = self.cam(cam_out)
    #     cam_out = self.cam2conv(cam_out)

    #     # segmentation result
    #     # 输出后续的结果分析时，两个注意力机制的输出值得研究
    #     # outputs = []
    #     # feats_sum = pam_out + cam_out
    #     # outputs.append(self.conv_out(feats_sum))
    #     # outputs.append(self.conv_pam_out(pam_out))
    #     # outputs.append(self.conv_cam_out(cam_out))

    #     # 修改后，只输出融合结果
    #     feats_sum = pam_out + cam_out
    #     output = self.conv_out(feats_sum)
    #     output = F.upsample_bilinear(output, (h, w))
    #     return output
    
    