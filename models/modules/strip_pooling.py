"""
@author: https://github.com/Andrew-Qibin/SPNet
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class StripPooling(nn.Module):
    """
    条纹池化：垂直和水平方向的池化，实际上是有条纹先验的注意力机制模块
    NOTE: 原始代码没有论文中的与输入的点乘加权
    """
    def __init__(self, inplanes, outplanes, norm_layer=None, identity=True):
        super(StripPooling, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False) # 3x1卷积用于调制当前位置及其邻近特征
        self.bn1 = norm_layer(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True) # 1x1卷积用于融合

        # 条纹池化
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

        self.identity = identity # 开启残差连接

    def forward(self, x):
        x_in = x
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        if self.identity:
            x = x_in * x
        return x


class MixedStripPooling(nn.Module):
    """
    混合条纹池化＋正常的方形池化
    """
    def __init__(self, in_channels, pool_size):
        """
        不改变通道数
        """
        super(MixedStripPooling, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = {
            "mode": 'bilinear',
            "align_corners": True
        }

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


###########################################################################
# Created by: luzhaoxin
# Email: luzhaoxin1998@gmail.com
# Copyright (c) 2022
# Reference: 基于深度学习的常见苹果叶片病害识别与病斑分割方法研究--晁晓菲，2021西北农林科技大学博士论文
###########################################################################

class DiagAvgPooling(nn.Module):
    """
    对主对角线指定偏移进行均值池化
    """
    def __init__(self, offset=0, dim1=2, dim2=3):
        """
        @param offset: 相对主对角线的偏移
        @param dim1: 获取对角线值的第一个维度，默认为2表示H
        @param dim2: 获取对角线值的第二个维度，默认为3表示W
        """
        super(DiagAvgPooling, self).__init__()
        self.offset = offset
        self.dim1 = dim1
        self.dim2 = dim2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x):
        x =  torch.diagonal(x, self.offset, dim1=self.dim1, dim2=self.dim2)
        x = self.avg_pool(x)
        return x

class DiagPooling(nn.Module):
    """
    对角线池化
    """
    def __init__(self, is_main_diag=True):
        """
        @param is_main_diag: 是否对主对角线进行池化, False时对次对角线进行池化
        """
        super(DiagPooling, self).__init__()
        self.is_main_diag = is_main_diag
    
    def forward(self, x):
        _, _, h, w = x.size()
        if self.is_main_diag:
            diag_list = [self._diag_avg_pooling(x, offset) for offset in range(-h//2, h//2 + 1, 1)]
        else:
            # 转置,将次对角线转换至主对角线
            x = torch.transpose(x, dim0=2, dim1=3)
            # 转换后的主对角线是逆序的
            diag_list = [self._diag_avg_pooling(x, offset) for offset in range(-h//2, h//2 + 1, 1)][::-1]
        x = torch.cat(diag_list, dim=-1) # 按照顺序拼接
        return x
    
    def _diag_avg_pooling(self, x, offset=0):
        """
        对主对角线指定偏移进行均值池化
        """
        x =  torch.diagonal(x, offset, dim1=2, dim2=3)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        return x
        


class MainDiagTwillPooling(nn.Module):
    """
    主对角线斜纹池化模块
    """
    def __init__(self,):
        super(MainDiagTwillPooling, self).__init__()


class SecondDiagTwillPooling(nn.Module):
    """
    次对角线斜纹池化模块
    """
    def __init__(self,):
        super(SecondDiagTwillPooling, self).__init__()


###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

# class SPNet(BaseNet):
#     def __init__(self, nclass, backbone, pretrained, criterion=None, aux=True, norm_layer=nn.BatchNorm2d, spm_on=False, **kwargs):
#         super(SPNet, self).__init__(nclass, backbone, aux, pretrained, norm_layer=norm_layer, spm_on=spm_on, **kwargs)
#         self.head = SPHead(2048, nclass, norm_layer, self._up_kwargs)
#         self.criterion = criterion
#         if aux:
#             self.auxlayer = FCNHead(1024, nclass, norm_layer)

#     def forward(self, x, y=None):
#         _, _, h, w = x.size()
#         _, _, c3, c4 = self.base_forward(x)

#         x = self.head(c4)
#         x = interpolate(x, (h,w), **self._up_kwargs)
#         if self.aux:
#             auxout = self.auxlayer(c3)
#             auxout = interpolate(auxout, (h,w), **self._up_kwargs)
        
#         if self.training:
#             aux = self.auxlayer(c3)
#             aux = interpolate(aux, (h, w), **self._up_kwargs)
#             main_loss = self.criterion(x, y)
#             aux_loss = self.criterion(aux, y)
#             return x.max(1)[1], main_loss, aux_loss
#         else:
#             return x


# class SPHead(nn.Module):
#     def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
#         super(SPHead, self).__init__()
#         inter_channels = in_channels // 2
#         self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
#                 norm_layer(inter_channels),
#                 nn.ReLU(True)
#         )
#         self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
#         self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
#         self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
#                 norm_layer(inter_channels // 2),
#                 nn.ReLU(True),
#                 nn.Dropout2d(0.1, False),
#                 nn.Conv2d(inter_channels // 2, out_channels, 1))

#     def forward(self, x):
#         x = self.trans_layer(x)
#         x = self.strip_pool1(x)
#         x = self.strip_pool2(x)
#         x = self.score_layer(x)
#         return x


if __name__ == "__main__":
    data = torch.Tensor([[[
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ]]])