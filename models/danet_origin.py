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
import torch.nn as nn
from itertools import chain
import numpy as np
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
from .drnet import drn_d_22, drn_d_38, drn_d_54
from .resnet import resnet50


from utils.helpers import initialize_weights, set_trainable

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




class DA_Net_(nn.Module):
    def __init__(self, out_channels=4, backbone="resnet50", in_channels=2048, pretrained=False, **_):
        super(DA_Net_, self).__init__()
        
        # set a base model
        if backbone == 'drn_d_22':
            print('Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=pretrained, num_classes=out_channels)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        elif backbone == 'drn_d_38':
            print('Dilated ResNet D 38 wil be used as a base model')
            self.base = drn_d_38(pretrained=pretrained, num_classes=out_channels)
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
            self.base = drn_d_22(pretrained=pretrained, num_classes=out_channels)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])

        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))


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
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)

        feats_sum = sc_output + sasc_output
        output = self.conv8(feats_sum)
        output = F.upsample_bilinear(output, (h, w))
        
        return output

        # # output = [sasc_output]
        # output = [sasc_output]
        # # output.append(sa_output)
        # output.append(sa_output)
        # # output.append(sc_output)
        # output.append(sc_output)
        # # return tuple(output)
        # return tuple(output)

        #修改后只输出融合结果
        # sasc_output = F.upsample_bilinear(sasc_output, (h, w))
        # return sasc_output
        # feats_sum = pam_out + cam_out
        # output = self.conv_out(feats_sum)
        # output = F.upsample_bilinear(output, (h, w))
        # return output

        # b, c, h, w = x.size()
        # x = self.base(x)

        # # outputs from attention modules
        # pam_out = self.conv2pam(x)
        # pam_out = self.pam(pam_out)
        # pam_out = self.pam2conv(pam_out)

        # cam_out = self.conv2cam(x)
        # cam_out = self.cam(cam_out)
        # cam_out = self.cam2conv(cam_out)

        # # segmentation result
        # # 输出后续的结果分析时，两个注意力机制的输出值得研究
        # # outputs = []
        # # feats_sum = pam_out + cam_out
        # # outputs.append(self.conv_out(feats_sum))
        # # outputs.append(self.conv_pam_out(pam_out))
        # # outputs.append(self.conv_cam_out(cam_out))

        # # 修改后，只输出融合结果
        # feats_sum = pam_out + cam_out
        # output = self.conv_out(feats_sum)
        # output = F.upsample_bilinear(output, (h, w))
        # return output
    
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