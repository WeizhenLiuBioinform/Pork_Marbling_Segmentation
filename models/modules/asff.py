"""
Learning Spatial Fusion for Single-Shot Object Detection 2019 cvpr
"""

import torch
import torch.nn as nn
import torchvision

def Conv1x1BnRelu(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU6(inplace=True),# 移动端，抑制大于6的值
        nn.ReLU(inplace=True)
    )

def upSampling1(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
        nn.Upsample(scale_factor=2, mode='nearest')
    )

def upSampling2(in_channels,out_channels):
    return nn.Sequential(
        upSampling1(in_channels,out_channels),
        nn.Upsample(scale_factor=2, mode='nearest'),
    )

def downSampling1(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def downSampling2(in_channels,out_channels):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
        downSampling1(in_channels=in_channels, out_channels=out_channels),
    )

# class ASFF(nn.Module):
#     def __init__(self, level, channel1, channel2, channel3, out_channel):
#         super(ASFF, self).__init__()
#         self.level = level
#         funsed_channel = 8

#         if self.level == 1:
#             # level = 1:
#             self.level2_1 = downSampling1(channel2,channel1)
#             self.level3_1 = downSampling2(channel3,channel1)

#             self.weight1 = Conv1x1BnRelu(channel1, funsed_channel)
#             self.weight2 = Conv1x1BnRelu(channel1, funsed_channel)
#             self.weight3 = Conv1x1BnRelu(channel1, funsed_channel)

#             self.expand_conv = Conv1x1BnRelu(channel1,out_channel)

#         if self.level == 2:
#             #  level = 2:
#             self.level1_2 = upSampling1(channel1,channel2)
#             self.level3_2 = downSampling1(channel3,channel2)

#             self.weight1 = Conv1x1BnRelu(channel2, funsed_channel)
#             self.weight2 = Conv1x1BnRelu(channel2, funsed_channel)
#             self.weight3 = Conv1x1BnRelu(channel2, funsed_channel)

#             self.expand_conv = Conv1x1BnRelu(channel2, out_channel)

#         if self.level == 3:
#             #  level = 3:
#             self.level1_3 = upSampling2(channel1,channel3)
#             self.level2_3 = upSampling1(channel2,channel3)

#             self.weight1 = Conv1x1BnRelu(channel3, funsed_channel)
#             self.weight2 = Conv1x1BnRelu(channel3, funsed_channel)
#             self.weight3 = Conv1x1BnRelu(channel3, funsed_channel)

#             self.expand_conv = Conv1x1BnRelu(channel3, out_channel)

#         self.weight_level = nn.Conv2d(funsed_channel * 3, 3, kernel_size=1, stride=1, padding=0)

#         self.softmax = nn.Softmax(dim=1)


#     def forward(self, x, y, z):
#         if self.level == 1:
#             level_x = x
#             level_y = self.level2_1(y)
#             level_z = self.level3_1(z)

#         if self.level == 2:
#             level_x = self.level1_2(x)
#             level_y = y
#             level_z = self.level3_2(z)

#         if self.level == 3:
#             level_x = self.level1_3(x)
#             level_y = self.level2_3(y)
#             level_z = z

#         weight1 = self.weight1(level_x)
#         weight2 = self.weight2(level_y)
#         weight3 = self.weight3(level_z)

#         level_weight = torch.cat((weight1, weight2, weight3), 1)
#         weight_level = self.weight_level(level_weight)
#         weight_level = self.softmax(weight_level)

#         fused_level = level_x * weight_level[:,0,:,:] + level_y * weight_level[:,1,:,:] + level_z * weight_level[:,2,:,:]
#         out = self.expand_conv(fused_level)
#         return out


class ASFF5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASFF5, self).__init__()
        self.weight_All = nn.Conv2d(in_channels*5, 5, kernel_size=1, stride=1, padding=0)
        self.weight1 = Conv1x1BnRelu(in_channels, out_channels)
        self.weight2 = Conv1x1BnRelu(in_channels, out_channels)
        self.weight3 = Conv1x1BnRelu(in_channels, out_channels)
        self.weight4 = Conv1x1BnRelu(in_channels, out_channels)
        self.weight5 = Conv1x1BnRelu(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)
        self.expand_conv = Conv1x1BnRelu(in_channels,out_channels)


    def forward(self, f1, f2, f3, f4, f5):
        w1 = self.weight1(f1)
        w2 = self.weight2(f2)
        w3 = self.weight3(f3)
        w4 = self.weight3(f4)
        w5 = self.weight3(f5)

        level_weight = torch.cat((w1, w2, w3, w4, w5), 1)
        weight_level = self.weight_All(level_weight)
        weight_level = self.softmax(weight_level)

        fused_level = f1 * weight_level[:,0:1,:,:] + f2 * weight_level[:,1:2,:,:] + f3 * weight_level[:,2:3,:,:] + f4 * weight_level[:,3:4,:,:] + f5 * weight_level[:,4:,:,:]
        fused_level = self.expand_conv(fused_level)
        return fused_level

class ASFF2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASFF2, self).__init__()
        self.weight_All = nn.Conv2d(in_channels*2, 2, kernel_size=1, stride=1, padding=0)
        self.weight1 = Conv1x1BnRelu(in_channels, out_channels)
        self.weight2 = Conv1x1BnRelu(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=1)
        self.expand_conv = Conv1x1BnRelu(in_channels,out_channels)


    def forward(self, f1, f2):
        w1 = self.weight1(f1)
        w2 = self.weight2(f2)

        level_weight = torch.cat((w1, w2), 1)
        weight_level = self.weight_All(level_weight)
        weight_level = self.softmax(weight_level)

        fused_level = f1 * weight_level[:,0:1,:,:] + f2 * weight_level[:,1:2,:,:]
        fused_level = self.expand_conv(fused_level)
        return fused_level

if __name__ == '__main__':
    model = ASFF5(256, 256)
    print(model)

    x = torch.randn(2, 256, 32, 32)
    y = torch.randn(2, 256, 32, 32)
    z = torch.randn(2, 256, 32, 32)
    o = torch.randn(2, 256, 32, 32)
    p = torch.randn(2, 256, 32, 32)
    out = model(x,y,z, o, p)
    print(out.shape)
