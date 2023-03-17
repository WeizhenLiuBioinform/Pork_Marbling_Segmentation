# from base import BaseModel
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from itertools import chain
# from base import BaseModel
# from utils.helpers import initialize_weights, set_trainable
# from itertools import chain
# from models import resnet


# def x2conv(in_channels, out_channels, inner_channels=None):
#     inner_channels = out_channels // 2 if inner_channels is None else inner_channels
#     down_conv = nn.Sequential(
#         nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(inner_channels),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True))
#     return down_conv

# class encoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(encoder, self).__init__()
#         self.down_conv = x2conv(in_channels, out_channels)
#         self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

#     def forward(self, x):
#         x = self.down_conv(x)
#         x = self.pool(x)
#         return x

# class decoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(decoder, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         self.up_conv = x2conv(in_channels, out_channels)

#     def forward(self, x_copy, x, interpolate=True):
#         x = self.up(x)

#         if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
#             if interpolate:
#                 # Iterpolating instead of padding
#                 x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
#                                 mode="bilinear", align_corners=True)
#             else:
#                 # Padding in case the incomping volumes are of different sizes
#                 diffY = x_copy.size()[2] - x.size()[2]
#                 diffX = x_copy.size()[3] - x.size()[3]
#                 x = F.pad(x, (diffX // 2, diffX - diffX // 2,
#                                 diffY // 2, diffY - diffY // 2))

#         # Concatenate
#         x = torch.cat([x_copy, x], dim=1)
#         x = self.up_conv(x)
#         return x


# class UNet(BaseModel):
#     def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
#         super(UNet, self).__init__()

#         self.start_conv = x2conv(in_channels, 64)
#         self.down1 = encoder(64, 128)
#         self.down2 = encoder(128, 256)
#         self.down3 = encoder(256, 512)
#         self.down4 = encoder(512, 1024)

#         self.middle_conv = x2conv(1024, 1024)

#         self.up1 = decoder(1024, 512)
#         self.up2 = decoder(512, 256)
#         self.up3 = decoder(256, 128)
#         self.up4 = decoder(128, 64)
#         self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
#         self._initialize_weights()

#         if freeze_bn:
#             self.freeze_bn()

#     def _initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     module.bias.data.zero_()
#             elif isinstance(module, nn.BatchNorm2d):
#                 module.weight.data.fill_(1)
#                 module.bias.data.zero_()

#     def forward(self, x):
#         x1 = self.start_conv(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.middle_conv(self.down4(x4))

#         x = self.up1(x4, x)
#         x = self.up2(x3, x)
#         x = self.up3(x2, x)
#         x = self.up4(x1, x)

#         x = self.final_conv(x)
#         return x

#     def get_backbone_params(self):
#         # There is no backbone for unet, all the parameters are trained from scratch
#         return []

#     def get_decoder_params(self):
#         return self.parameters()

#     def freeze_bn(self):
#         for module in self.modules():
#             if isinstance(module, nn.BatchNorm2d): module.eval()




# """
# -> Unet with a resnet backbone
# """

# class UNetResnet(BaseModel):
#     def __init__(self, num_classes, in_channels=3, backbone='resnet34', pretrained=True, freeze_bn=False, freeze_backbone=False, **_):
#         super(UNetResnet, self).__init__()
#         model = getattr(resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d)
#         expansion = model.expansion # resnet不同block的扩张率不同
#         self.initial = list(model.children())[:4]
#         if in_channels != 3:
#             self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.initial = nn.Sequential(*self.initial)

#         # encoder
#         self.layer1 = model.layer1
#         self.layer2 = model.layer2
#         self.layer3 = model.layer3
#         self.layer4 = model.layer4

#         # decoder
#         self.conv1 = nn.Conv2d(512*expansion, 192, kernel_size=3, stride=1, padding=1)
#         self.upconv1 =  nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)

#         self.conv2 = nn.Conv2d(256*expansion + 128, 128, kernel_size=3, stride=1, padding=1)
#         self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)

#         self.conv3 = nn.Conv2d(128*expansion + 96, 96, kernel_size=3, stride=1, padding=1)
#         self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)

#         self.conv4 = nn.Conv2d(64*expansion + 64, 64, kernel_size=3, stride=1, padding=1)
#         self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)
        
#         self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
#         self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)

#         self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

#         initialize_weights(self)

#         if freeze_bn:
#             self.freeze_bn()
#         if freeze_backbone: 
#             set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
#         x1 = self.layer1(self.initial(x))
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
        
#         x = self.upconv1(self.conv1(x4))
#         x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True)
#         x = torch.cat([x, x3], dim=1)
#         x = self.upconv2(self.conv2(x))

#         x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
#         x = torch.cat([x, x2], dim=1)
#         x = self.upconv3(self.conv3(x))

#         x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
#         x = torch.cat([x, x1], dim=1)

#         x = self.upconv4(self.conv4(x))

#         x = self.upconv5(self.conv5(x))

#         # if the input is not divisible by the output stride
#         if x.size(2) != H or x.size(3) != W:
#             x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

#         x = self.conv7(self.conv6(x))
#         return x

#     def get_backbone_params(self):
#         return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
#                     self.layer3.parameters(), self.layer4.parameters())

#     def get_decoder_params(self):
#         return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(), self.upconv2.parameters(),
#                     self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(), self.upconv4.parameters(),
#                     self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(), self.conv7.parameters())

#     def freeze_bn(self):
#         for module in self.modules():
#             if isinstance(module, nn.BatchNorm2d): module.eval()


from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
from models import resnet
from functools import partial

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
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

def x2conv(in_channels, out_channels, inner_channels=None):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.pool(x)
        return x

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=False)
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x

class decoder1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder1, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = x2conv(1024, out_channels)

    def forward(self, x_copy, x, interpolate=True):#x_copy:512,50,100   x:#1028,25,50
        x = self.up(x)#512,50,100 

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=False)
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x

class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()

        self.start_conv = x2conv(in_channels, 64)
        self.down1 = encoder(64, 128)
        self.down2 = encoder(128, 256)
        self.down3 = encoder(256, 512)
        self.down4 = encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        # self.dblock = DACblock(1024)
        # self.spp = SPPblock(1024)
                                                                                                                            
        self.up_1 = decoder1(1028, 512)

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None: 
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        #x = F.upsample(x, size=(2*x.size(2),2*x.size(3)), mode='bilinear')
        x1 = self.start_conv(x)#64,400,800
        x2 = self.down1(x1)#128,200,400
        x3 = self.down2(x2)#256,100,200
        x4 = self.down3(x3)#512,50,100
        x = self.down4(x4)#1024,25,50
        x = self.middle_conv(x)#1024,25,50
        
        #多尺度特征提取
        # x = self.dblock(x)
        # x = self.spp(x)#1028,25,50
        # x = self.up_1(x4,x)#这个是1028个通道对应的decoder
        
        x = self.up1(x4, x)#这个是1024个通道对应的decoder
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        #x = self.maxpool(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()




"""
-> Unet with a resnet backbone
"""

class UNetResnet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet34', pretrained=True, freeze_bn=False, freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        model = getattr(resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d)
        expansion = model.expansion # resnet不同block的扩张率不同
        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # decoder
        self.conv1 = nn.Conv2d(512*expansion, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 =  nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(256*expansion + 128, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)

        self.conv3 = nn.Conv2d(128*expansion + 96, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(64*expansion + 64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)
        
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

        initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))

        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))

        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)

        x = self.upconv4(self.conv4(x))

        x = self.upconv5(self.conv5(x))

        # if the input is not divisible by the output stride
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.conv7(self.conv6(x))
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                    self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(), self.upconv2.parameters(),
                    self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(), self.upconv4.parameters(),
                    self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(), self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
