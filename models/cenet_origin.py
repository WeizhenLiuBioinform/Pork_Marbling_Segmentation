import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from itertools import chain
# import os
# import sys
# sys.path.append('/home/cr/wokspace/vein_code/psp_test')
from functools import partial
# from models.attention import Attention2d
# from models.ca import CoordAtt
from models.da_att import PAM_Module,CAM_Module
# from models.epsanet import PSAModule
# from models.STLNet import TEM 
# from models.STLNet import PTFEM 
# from models.STLNet import ConvBNReLU 
# import Constants
# # from networks.ccnet import RCCAModule

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


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
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


class DACblock_with_inception(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class DACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super(DecoderBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nonlinearity

#         self.deconv1 = nn.ConvTranspose2d(in_channels // 4, in_channels // 16, (1, 9), stride=2, padding=(0, 4), output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 16, (9, 1), stride=2, padding=(4, 0), output_padding=1)
#         self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 16, (9, 1), padding=(4, 0))
#         self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 16, (1, 9), padding=(0, 4))
#         # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, (1,9), stride=2, padding=(0,4), output_padding=1)
#         if in_channels == 516:
#             self.norm2 = nn.BatchNorm2d(128)
#             self.conv3 = nn.Conv2d(128, n_filters, 1)
#         else:
#             self.norm2 = nn.BatchNorm2d(in_channels // 4)
#             self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        
#         self.relu2 = nonlinearity

#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nonlinearity
#     def forward(self, x, inp = False):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)

#         x1 = self.deconv1(x)
#         x2 = self.deconv2(x)
#         d1 = self.h_transform(x)#(2,129,32,63)
#         d2 = self.deconv3(d1)#(2,32,32,63)
#         d3 = self.inv_h_transform(d2)
#         x3 = F.interpolate(d3, scale_factor=2)
#         e1 = self.v_transform(x)#(2, 129, 32, 32)
#         e2 = self.deconv4(e1)
#         e3 = self.inv_v_transform(e2)
#         x4 = F.interpolate(e3, scale_factor=2)
#         x = torch.cat((x1, x2, x3, x4), 1)
        
#         # x = F.interpolate(x, scale_factor=2)
#         x = self.norm2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x

#     def h_transform(self, x):#(2,129,32,32)
#         shape = x.size()
#         x = torch.nn.functional.pad(x, (0, shape[-1]))#(2,129,32,64)
#         x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]#(2,129,2016)
#         x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)#(2,129,32,63)
#         return x

#     def inv_h_transform(self, x):#(2,32,32,63)
#         shape = x.size()
#         x = x.reshape(shape[0], shape[1], -1).contiguous()#(2,32,2016)
#         x = torch.nn.functional.pad(x, (0, shape[-2]))#(2,32,2048)
#         x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])#(2,32,32,64)
#         x = x[..., 0: shape[-2]]#(2,32,32,32)
#         return x

#     def v_transform(self, x):
#         x = x.permute(0, 1, 3, 2)
#         shape = x.size()
#         x = torch.nn.functional.pad(x, (0, shape[-1]))
#         x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
#         x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
#         return x.permute(0, 1, 3, 2)

#     def inv_v_transform(self, x):
#         x = x.permute(0, 1, 3, 2)
#         shape = x.size()
#         x = x.reshape(shape[0], shape[1], -1)
#         x = torch.nn.functional.pad(x, (0, shape[-2]))
#         x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
#         x = x[..., 0: shape[-2]]
#         return x.permute(0, 1, 3, 2)
class Decoder(nn.Module):
    def __init__(self,filters):
        super(Decoder, self).__init__()

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


    def forward(self, e1, e2, e3, e4):
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        return d1


class ASPP(nn.Module):
    def __init__(self, in_channel=2048):
        depth = in_channel
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net

class CE_Net_(nn.Module):
    def __init__(self, num_classes=4, in_channels=3, **_):#backbone='resnet34', pretrained=False, output_stride=16, freeze_bn=False, freeze_backbone=False,
        super(CE_Net_, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        # resnet.train = True
        # state_dict = torch.load(r"/home/data/workspace/cr_vein/pretrained/resnet34-333f7ec4.pth")
        # resnet.load_state_dict(state_dict)
        self.aspp4 = ASPP(in_channel=516)
        self.aspp3 = ASPP(in_channel=256)
        self.aspp2 = ASPP(in_channel=128)
        self.aspp1 = ASPP(in_channel=64)
        self.aspp = ASPP(in_channel=516)
        self.firstconv = resnet.conv1
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        # att 注意力机制 引入 da 两个注意力机制
        self.ca1 = CAM_Module(filters[0])
        self.pa1 = PAM_Module(filters[0])
        # self.att1 =  Attention2d(filters[0])
        self.encoder2 = resnet.layer2
        self.ca2 = CAM_Module(filters[1])
        self.pa2 = PAM_Module(filters[1])
        
        # self.att2 =  Attention2d(filters[1])
        self.encoder3 = resnet.layer3
        self.ca3 = CAM_Module(filters[2])
        self.pa3 = PAM_Module(filters[2])
        
        # self.att3 =  Attention2d(filters[2])
        self.encoder4 = resnet.layer4
        self.ca4 = CAM_Module(filters[3])
        self.pa4 = PAM_Module(filters[3])
        # 增加十字交叉注意力机制
        # self.cca3 = RCCAModule(256,512,256)

        # self.cca4 = RCCAModule(512,512,512)
        #增加多尺度注意力机制
        #增加纹理增强模块
        # self.conv_start = ConvBNReLU(filters[0], 256, 1, 1, 0)
        # self.tem = TEM(filters[0])
        # self.ptfem = PTFEM()
        # self.maxpool2d = nn.MaxPool2d(8,8)
        # self.con2d = nn.Conv2d(512,256,(1,1))
        # conv_kernels=[3, 5, 7, 9]
        # conv_groups=[1, 4, 8, 16]
        # planes = 0
        # self.psa = PSAModule(planes, planes ,stride=1, conv_kernels=conv_kernels, conv_groups=conv_groups)
  

        # self.att4 =  Attention2d(filters[3])

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder = Decoder(filters)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):

        # Encoder
        # print("x:"+str(x.size()))

        x = self.firstconv(x)
        # print("firstconv(x):"+str(x.size()))

        x = self.firstbn(x)
        # print("firstbn(x):"+str(x.size()))

        x = self.firstrelu(x)
        # print("firstrelu(x):"+str(x.size()))

        x = self.firstmaxpool(x)
        # print("firstmaxpool(x):"+str(x.size()))

        e1 = self.encoder1(x)
        # e1 = self.aspp1(e1)
        
        # print("e1:"+str(e1.size()))
        # e1 = self.att1(e1)
        # ca1 = self.ca1(e1)
        # pa1 = self.pa1(e1)
        # e1 = ca1 + pa1

        e2 = self.encoder2(e1)
        # e2 = self.aspp2(e2)
        # print("e2:"+str(e2.size()))
        # e2 = self.att2(e2)
        # ca2 = self.ca2(e2)
        # pa2 = self.pa2(e2)
        # e2 = ca2 + pa2
        e3 = self.encoder3(e2)
        # ca3 = self.ca3(e3)
        # pa3 = self.pa3(e3)
        # e3 = ca3 + pa3
        # e3 = self.aspp3(e3)
        # print("e3:"+str(e3.size()))
        # e3 = self.cca3(e3)
        e4 = self.encoder4(e3)
        
        # e4 = self.cca4(e4)
        # print("e4:"+str(e4.size()))
        # ca4 = self.ca4(e4)
        # pa4 = self.pa4(e4)
        # e4 = ca4 + pa4
        # print("e4:"+str(e4.size()))

        # e2 = self.encoder2(e1) 
        # e3 = self.encoder3(e2)
        # e4 = self.encoder4(e3)
        # print("e4 加注意力后:"+str(e4.size()))
        # Center
        #增加纹理增强
        # x_zq = self.conv_start(e1)
        # x_tem = self.tem(x_zq)
        # x_zq = torch.cat([x_tem, x_zq], dim=1) #c = 256 + 256 = 512
        # x_ptfem = self.ptfem(x_zq) # 256   
        # x_zq = self.con2d(x_zq)
        # x_zq = torch.cat([x_ptfem, x_zq], dim=1)
        # x_zq = torch.cat([x_zq, e4], dim=)
        # x_zq = self.maxpool2d(x_zq)
        # e4 = e4 + x_zq
        e4 = self.dblock(e4)
        # print("e4 dblock:"+str(e4.size()))
        e4 = self.spp(e4)
        # print("e4 spp:"+str(e4.size()))
        # e4 = self.aspp4(e4)
        # e4 = self.aspp(e4)
        
        
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        # d1 = self.decoder(e1, e2, e3, e4)
        
        
        
        
        out = self.finaldeconv1(d1)
        # print("out finaldeconv1:"+str(out.size()))

        out = self.finalrelu1(out)
        # print("out finalrelu1:"+str(out.size()))

        out = self.finalconv2(out)
        # print("out finalconv2:"+str(out.size()))

        out = self.finalrelu2(out)
        # print("out finalrelu2:"+str(out.size()))

        out = self.finalconv3(out)
        # print("out finalconv3:"+str(out.size()))


        # return F.sigmoid(out)
        #return torch.nn.Sigmoid()(out)
        return out
    def get_backbone_params(self):
        return chain(self.parameters())

    def get_decoder_params(self):
        return []

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
    
    











# #这个是原始不加任何模块的cenet
# import torch
# import torch.nn as nn
# from torchvision import models
# import torch.nn.functional as F
# from itertools import chain
# # import os
# # import sys
# # sys.path.append('/home/cr/wokspace/vein_code/psp_test')
# from functools import partial
# # from models.attention import Attention2d
# # from models.ca import CoordAtt
# from models.da_att import PAM_Module,CAM_Module
# # from models.epsanet import PSAModule
# # from models.STLNet import TEM 
# # from models.STLNet import PTFEM 
# # from models.STLNet import ConvBNReLU 
# # import Constants
# # # from networks.ccnet import RCCAModule

# nonlinearity = partial(F.relu, inplace=True)


# class DACblock(nn.Module):
#     def __init__(self, channel):
#         super(DACblock, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
#         self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
#         self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
#         dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
#         dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
#         return out


# class DACblock_without_atrous(nn.Module):
#     def __init__(self, channel):
#         super(DACblock_without_atrous, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
#         dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
#         dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

#         return out


# class DACblock_with_inception(nn.Module):
#     def __init__(self, channel):
#         super(DACblock_with_inception, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

#         self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
#         dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
#         dilate3_out = nonlinearity(self.dilate1(dilate_concat))
#         out = x + dilate3_out
#         return out


# class DACblock_with_inception_blocks(nn.Module):
#     def __init__(self, channel):
#         super(DACblock_with_inception_blocks, self).__init__()
#         self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
#         self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
#         self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         dilate1_out = nonlinearity(self.conv1x1(x))
#         dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
#         dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
#         dilate4_out = self.pooling(x)
#         out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
#         return out


# class PSPModule(nn.Module):
#     def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
#         super().__init__()
#         self.stages = []
#         self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
#         self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
#         self.relu = nn.ReLU()

#     def _make_stage(self, features, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
#         return nn.Sequential(prior, conv)

#     def forward(self, feats):
#         h, w = feats.size(2), feats.size(3)
#         priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
#         bottle = self.bottleneck(torch.cat(priors, 1))
#         return self.relu(bottle)


# class SPPblock(nn.Module):
#     def __init__(self, in_channels):
#         super(SPPblock, self).__init__()
#         self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
#         self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
#         self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

#     def forward(self, x):
#         self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
#         self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
#         self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
#         self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
#         self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)
        
#         out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        
#         return out


# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super(DecoderBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nonlinearity

#         self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
#         # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, (1,9), stride=2, padding=(0,4), output_padding=1)
#         self.norm2 = nn.BatchNorm2d(in_channels // 4)
#         self.relu2 = nonlinearity

#         self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nonlinearity

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)
#         x = self.deconv2(x)
#         x = self.norm2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x
# # class DecoderBlock(nn.Module):
# #     def __init__(self, in_channels, n_filters):
# #         super(DecoderBlock, self).__init__()

# #         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
# #         self.norm1 = nn.BatchNorm2d(in_channels // 4)
# #         self.relu1 = nonlinearity

# #         self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 16, (1, 9), padding=(0, 4))
# #         self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 16, (9, 1), padding=(4, 0))
# #         self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 16, (9, 1), padding=(4, 0))
# #         self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 16, (1, 9), padding=(0, 4))
# #         # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, (1,9), stride=2, padding=(0,4), output_padding=1)
# #         if in_channels == 516:
# #             self.norm2 = nn.BatchNorm2d(128)
# #             self.conv3 = nn.Conv2d(128, n_filters, 1)
# #         else:
# #             self.norm2 = nn.BatchNorm2d(in_channels // 4)
# #             self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        
# #         self.relu2 = nonlinearity

# #         self.norm3 = nn.BatchNorm2d(n_filters)
# #         self.relu3 = nonlinearity
# #     def forward(self, x, inp = False):
# #         x = self.conv1(x)
# #         x = self.norm1(x)
# #         x = self.relu1(x)

# #         x1 = self.deconv1(x)
# #         x2 = self.deconv2(x)
# #         x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
# #         x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
# #         x = torch.cat((x1, x2, x3, x4), 1)
        
# #         x = F.interpolate(x, scale_factor=2)
# #         x = self.norm2(x)
# #         x = self.relu2(x)
# #         x = self.conv3(x)
# #         x = self.norm3(x)
# #         x = self.relu3(x)
# #         return x

# #     def h_transform(self, x):
# #         shape = x.size()
# #         x = torch.nn.functional.pad(x, (0, shape[-1]))
# #         x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
# #         x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
# #         return x

# #     def inv_h_transform(self, x):
# #         shape = x.size()
# #         x = x.reshape(shape[0], shape[1], -1).contiguous()
# #         x = torch.nn.functional.pad(x, (0, shape[-2]))
# #         x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
# #         x = x[..., 0: shape[-2]]
# #         return x

# #     def v_transform(self, x):
# #         x = x.permute(0, 1, 3, 2)
# #         shape = x.size()
# #         x = torch.nn.functional.pad(x, (0, shape[-1]))
# #         x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
# #         x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
# #         return x.permute(0, 1, 3, 2)

# #     def inv_v_transform(self, x):
# #         x = x.permute(0, 1, 3, 2)
# #         shape = x.size()
# #         x = x.reshape(shape[0], shape[1], -1)
# #         x = torch.nn.functional.pad(x, (0, shape[-2]))
# #         x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
# #         x = x[..., 0: shape[-2]]
# #         return x.permute(0, 1, 3, 2)
# # class Decoder(nn.Module):
# #     def __init__(self,filters):
# #         super(Decoder, self).__init__()

# #         self.decoder4 = DecoderBlock(516, filters[2])
# #         self.decoder3 = DecoderBlock(filters[2], filters[1])
# #         self.decoder2 = DecoderBlock(filters[1], filters[0])
# #         self.decoder1 = DecoderBlock(filters[0], filters[0])


# #     def forward(self, e1, e2, e3, e4):
# #         d4 = self.decoder4(e4) + e3
# #         d3 = self.decoder3(d4) + e2
# #         d2 = self.decoder2(d3) + e1
# #         d1 = self.decoder1(d2)

# #         return d1


# class ASPP(nn.Module):
#     def __init__(self, in_channel=2048):
#         depth = 516
#         super(ASPP,self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv2d(in_channel, depth, 1, 1)
#         self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
#         self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
#         self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
#         self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
#     def forward(self, x):
#         size = x.shape[2:]
 
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.interpolate(image_features, size=size, mode='bilinear')
 
#         atrous_block1 = self.atrous_block1(x)
#         atrous_block6 = self.atrous_block6(x)
#         atrous_block12 = self.atrous_block12(x)
#         atrous_block18 = self.atrous_block18(x)
 
#         cat = torch.cat([image_features, atrous_block1, atrous_block6,
#                          atrous_block12, atrous_block18], dim=1)
#         net = self.conv_1x1_output(cat)
#         return net

# class CE_Net_(nn.Module):
#     def __init__(self, num_classes=4, in_channels=3, **_):#backbone='resnet34', pretrained=False, output_stride=16, freeze_bn=False, freeze_backbone=False,
#         super(CE_Net_, self).__init__()

#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=False)
#         # resnet.train = True
#         # state_dict = torch.load(r"/home/data/workspace/cr_vein/pretrained/resnet34-333f7ec4.pth")
#         # resnet.load_state_dict(state_dict)
#         self.aspp = ASPP(in_channel=516)
#         self.firstconv = resnet.conv1
        
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         # att 注意力机制 引入 da 两个注意力机制
#         self.ca1 = CAM_Module(filters[0])
#         self.pa1 = PAM_Module(filters[0])
#         # self.att1 =  Attention2d(filters[0])
#         self.encoder2 = resnet.layer2
#         self.ca2 = CAM_Module(filters[1])
#         self.pa2 = PAM_Module(filters[1])
        
#         # self.att2 =  Attention2d(filters[1])
#         self.encoder3 = resnet.layer3
#         self.ca3 = CAM_Module(filters[2])
#         self.pa3 = PAM_Module(filters[2])
        
#         # self.att3 =  Attention2d(filters[2])
#         self.encoder4 = resnet.layer4
#         self.ca4 = CAM_Module(filters[3])
#         self.pa4 = PAM_Module(filters[3])
#         # 增加十字交叉注意力机制
#         # self.cca3 = RCCAModule(256,512,256)

#         # self.cca4 = RCCAModule(512,512,512)
#         #增加多尺度注意力机制
#         #增加纹理增强模块
#         # self.conv_start = ConvBNReLU(filters[0], 256, 1, 1, 0)
#         # self.tem = TEM(filters[0])
#         # self.ptfem = PTFEM()
#         # self.maxpool2d = nn.MaxPool2d(8,8)
#         # self.con2d = nn.Conv2d(512,256,(1,1))
#         # conv_kernels=[3, 5, 7, 9]
#         # conv_groups=[1, 4, 8, 16]
#         # planes = 0
#         # self.psa = PSAModule(planes, planes ,stride=1, conv_kernels=conv_kernels, conv_groups=conv_groups)
  

#         # self.att4 =  Attention2d(filters[3])

#         self.dblock = DACblock(512)
#         self.spp = SPPblock(512)

#         self.decoder4 = DecoderBlock(516, filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#         # self.decoder = Decoder(filters)

#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

#     def forward(self, x):

#         # Encoder
#         # print("x:"+str(x.size()))

#         x = self.firstconv(x)
#         # print("firstconv(x):"+str(x.size()))

#         x = self.firstbn(x)
#         # print("firstbn(x):"+str(x.size()))

#         x = self.firstrelu(x)
#         # print("firstrelu(x):"+str(x.size()))

#         x = self.firstmaxpool(x)
#         # print("firstmaxpool(x):"+str(x.size()))

#         e1 = self.encoder1(x)
        
#         # print("e1:"+str(e1.size()))
#         # e1 = self.att1(e1)
#         # ca1 = self.ca1(e1)
#         # pa1 = self.pa1(e1)
#         # e1 = ca1 + pa1

#         e2 = self.encoder2(e1)
#         # print("e2:"+str(e2.size()))
#         # e2 = self.att2(e2)
#         # ca2 = self.ca2(e2)
#         # pa2 = self.pa2(e2)
#         # e2 = ca2 + pa2
#         e3 = self.encoder3(e2)
#         # ca3 = self.ca3(e3)
#         # pa3 = self.pa3(e3)
#         # e3 = ca3 + pa3
#         # print("e3:"+str(e3.size()))
#         # e3 = self.cca3(e3)
#         e4 = self.encoder4(e3)
#         # e4 = self.cca4(e4)
#         # print("e4:"+str(e4.size()))
#         # ca4 = self.ca4(e4)
#         # pa4 = self.pa4(e4)
#         # e4 = ca4 + pa4
#         # print("e4:"+str(e4.size()))

#         # e2 = self.encoder2(e1) 
#         # e3 = self.encoder3(e2)
#         # e4 = self.encoder4(e3)
#         # print("e4 加注意力后:"+str(e4.size()))
#         # Center
#         #增加纹理增强
#         # x_zq = self.conv_start(e1)
#         # x_tem = self.tem(x_zq)
#         # x_zq = torch.cat([x_tem, x_zq], dim=1) #c = 256 + 256 = 512
#         # x_ptfem = self.ptfem(x_zq) # 256   
#         # x_zq = self.con2d(x_zq)
#         # x_zq = torch.cat([x_ptfem, x_zq], dim=1)
#         # x_zq = torch.cat([x_zq, e4], dim=)
#         # x_zq = self.maxpool2d(x_zq)
#         # e4 = e4 + x_zq
#         e4 = self.dblock(e4)
#         # print("e4 dblock:"+str(e4.size()))
#         e4 = self.spp(e4)
#         # print("e4 spp:"+str(e4.size()))

#         # e4 = self.aspp(e4)
        
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         # d1 = self.decoder(e1, e2, e3, e4)

        
        
        
#         out = self.finaldeconv1(d1)
#         # print("out finaldeconv1:"+str(out.size()))

#         out = self.finalrelu1(out)
#         # print("out finalrelu1:"+str(out.size()))

#         out = self.finalconv2(out)
#         # print("out finalconv2:"+str(out.size()))

#         out = self.finalrelu2(out)
#         # print("out finalrelu2:"+str(out.size()))

#         out = self.finalconv3(out)
#         # print("out finalconv3:"+str(out.size()))


#         # return F.sigmoid(out)
#         #return torch.nn.Sigmoid()(out)
#         return out
#     def get_backbone_params(self):
#         return chain(self.parameters())

#     def get_decoder_params(self):
#         return []

#     def freeze_bn(self):
#         for module in self.modules():
#             if isinstance(module, nn.BatchNorm2d): module.eval()
    
    

