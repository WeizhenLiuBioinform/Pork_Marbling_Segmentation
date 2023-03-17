import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from .vgg import vgg16_features

class RSB(nn.Module):
    """
    Residual Skip Block
    """
    def __init__(self, in_channels, filters, k_size=3, stride=1, skip=True):
        """
        @param in_channels:
        @param filters:
        """
        filter_1, filter_2, filter_3, filter_4 = filters
        self.seq1 = nn.Sequence(
            nn.Conv2d(in_channels, filter_1, (1, 1), stride=stride),
            nn.BatchNorm2d(filter_1, momentum=0.8),
            nn.ReLU(inplace=True)
        )
        self.seq2 = nn.Sequence(
            nn.Conv2d(filter_1, filter_2, (k_size, k_size), stride=stride),
            nn.BatchNorm2d(filter_2, momentum=0.8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(filter_2, filter_3, (1, 1), stride=stride)
        self.bn3 = nn.BatchNorm2d(filter_3, momentum=0.8)
        
        self.shortcut = nn.Sequential()
        if not skip:
            self.shortcut = nn.Sequential(
                nn.Conv2d(filter_3, filter_4, (1,1), stride=stride),
                nn.BatchNorm2d(filter_4, momentum=0.8)
            )
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        res = x
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        res = self.shortcut(res)
        x += res
        x = self.relu3(x)

class SuimEncoderRSB(nn.Module):
    """
    SUIM-Net Encoder
    320x240x3(W*H*C)
        ||
        \/ conv5x5
    316x236x64
        ||
        \/ maxpool3x3
    
    """
    out_channels_list = [
        64, 128, 256
    ]
    def __init__(self, in_channels):
        out_c_1, out_c_2, out_c_3 = SuimEncoderRSB.out_channels_list
        # encoder block 1
        self.conv = nn.Conv2d(in_channels, out_c_1, (5,5), stride=1)
        # encoder block 2
        self.bn = nn.BatchNorm2d(out_c_1, momentum=0.8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((3,3), stride=2)
        self.RSB_seq_1 = nn.Sequential(
            RSB(out_c_1, [out_c_1, out_c_1, out_c_2, out_c_2], 3, stride=2, skip=False),
            RSB(out_c_2, [out_c_1, out_c_1, out_c_2, out_c_2], 3, 1, skip=True),
            RSB(out_c_2, [out_c_1, out_c_1, out_c_2, out_c_2], 3, 1, skip=True),
        )
        # encoder block 3
        self.RSB_seq_2 = nn.Sequential(
            RSB(out_c_2, [out_c_2, out_c_2, out_c_3, out_c_3], 3, 2, False),
            RSB(out_c_3, [out_c_2, out_c_2, out_c_3, out_c_3], 3, 1, True),
            RSB(out_c_3, [out_c_2, out_c_2, out_c_3, out_c_3], 3, 1, True),
            RSB(out_c_3, [out_c_2, out_c_2, out_c_3, out_c_3], 3, 1, True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        enc_1 = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.RSB_seq_1(x)
        enc_2 = x
        x = self.RSB_seq_2(x)
        enc_3 = x
        return enc_1, enc_2, enc_3

class SkipConcat(nn.Module):
    """
    合并来自编码器的跳层连接
    """
    def __init__(self,in_channels, out_channels, k_size=3):
        self.conv = nn.Conv2d(in_channels, out_channels, k_size, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.8)
    
    def forward(self, layer_input, skip_input):
        x = self.conv(layer_input)
        x = self.relu(x)
        x = self.bn(x)
        x = torch.cat([x, skip_input], dim=1) # 沿通道拼接
        return x

# class SuimDecoderRSB(nn.Module):
#     """
#     SUIM-Net Decoder
#     """
#     out_channels_list = [256, 128, 64]
#     def __init__(self, in_channels, num_channels, encoder_out_channels=[]):
#         enc_out_c_1, enc_out_c_2, enc_out_c_3 = SuimEncoderRSB.out_channels_list
#         dec_out_c_1, dec_out_c_2, dec_out_c_3 = SuimDecoderRSB.out_channels_list
#         self.dec_seq_1 = nn.Sequential(
#             nn.Conv2d(enc_out_c_3, dec_out_c_1, (3,3), 1, 1),
#             nn.BatchNorm2d(dec_out_c_1, momentum=0.8),
            
#         )

class MyDeconv2D(nn.Module):
    """
    反卷积上采样
    """
    def __init__(self, in_channels,out_channels, k_size=3):
        super(MyDeconv2D, self).__init__()
        # self.deconv = nn.ConvTranspose2d(in_channels, out_channels, k_size, stride=2, padding=1)
        # 反卷积在kernelsize为奇数时会出现棋盘格效应
        self.up2x = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.8)
    
    def forward(self, layer_input, skip_input):
        x = self.up2x(layer_input)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = torch.cat([x, skip_input], dim=1) # 按通道叠加
        return x


class SuimVGG16(nn.Module):
    """
    SUIM-Net VGG16
    """
    def __init__(self, num_classes, in_channels=3, pretrained=True, **_):
        super(SuimVGG16, self).__init__()
        self.features = vgg16_features(pretrained=pretrained, in_channels=in_channels, num_classes=num_classes)
        outc_1, outc_2, outc_3, outc_4 = self.features.out_channels_list
        self.block1 = self.features.block1
        self.block2 = self.features.block2
        self.block3 = self.features.block3
        self.block4 = self.features.block4
        # decoder
        self.dec1 = MyDeconv2D(outc_4, 512)
        self.dec2 = MyDeconv2D(outc_3+512, 256)
        self.dec3 = MyDeconv2D(outc_2+256, 128)
        self.up2x = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv4 = nn.Conv2d(128+outc_1, num_classes, (3,3), 1, 1)
    
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        # decoder
        x = self.dec1(x4, x3)
        x = self.dec2(x, x2)
        x = self.dec3(x, x1)
        x = self.up2x(x)
        x = self.conv4(x)
        return x
    
    def get_backbone_params(self):
        return chain(self.block1.parameters(), self.block2.parameters(), self.block3.parameters(), self.block4.parameters())

    def get_decoder_params(self):
        return chain(self.dec1.parameters(), self.dec2.parameters(), self.dec3.parameters(), self.conv4.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()