import torch
import torch.nn as nn

from torch.nn import functional as F
 
class BasicBlock(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
        base_width = 64, dilation = 1, norm_layer = None):
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample= None,
        groups = 1, base_width = 64, dilation = 1, norm_layer = None,):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(
        self,block, layers,num_classes = 4, zero_init_residual = False, groups = 1,
        width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
 
    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = stride
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,  planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
 
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
 
    def _forward_impl(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        return outs
 
    def forward(self, x) :
        return self._forward_impl(x)
    def _resnet(block, layers, pretrained_path = None, **kwargs,):
        model = ResNet(block, layers, **kwargs)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path),  strict=False)
        return model
    
    def resnet50(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 6, 3],pretrained_path,**kwargs)
    
    def resnet101(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 23, 3],pretrained_path,**kwargs)


 
class Encoding(nn.Module):
 
    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels)**0.5)
        # [num_codes, channels]
        self.codewords = nn.Parameter(
            torch.empty(num_codes, channels,
                        dtype=torch.float).uniform_(-std, std),
            requires_grad=True)
        # [num_codes]
        self.scale = nn.Parameter(
            torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0),
            requires_grad=True)
        
    def scaled_l2(self, x, codewords, scale):
        num_codes, channels = codewords.size()
        batch_size = x.size(0)
        reshaped_scale = scale.view((1, 1, num_codes))
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
 
        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm
 
 
    def aggregate(self, assigment_weights, x, codewords):
        num_codes, channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))
        batch_size = x.size(0)
 
        expanded_x = x.unsqueeze(2).expand(
            (batch_size, x.size(1), num_codes, channels))
        encoded_feat = (assigment_weights.unsqueeze(3) *
                        (expanded_x - reshaped_codewords)).sum(dim=1)
        return encoded_feat
 
    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.channels
        # [batch_size, channels, height, width]
        batch_size = x.size(0)
        # [batch_size, height x width, channels]
        x = x.view(batch_size, self.channels, -1).transpose(1, 2).contiguous()
 
        # assignment_weights: [batch_size, channels, num_codes]
        assigment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        # aggregate
        #print("assigment_weights:",assigment_weights.shape)
        encoded_feat = self.aggregate(assigment_weights, x, self.codewords)
        #print("encoded_feat1:",encoded_feat.shape)
        return encoded_feat
 
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}' \
                    f'x{self.channels})'
        return repr_str

 
class EncModule(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(EncModule, self).__init__()
        self.encoding_project = nn.Conv2d(
            in_channels,
            in_channels,
            1,
            )
        # TODO: resolve this hack
        # change to 1d
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels), nn.Sigmoid())
 
    def forward(self, x):
        """Forward function."""
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        
        #print("encoding_feat2: ",encoding_feat.shape)
        
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output
 
 
 
class EncHead(nn.Module):
 
    def __init__(self,num_classes=2,
                 num_codes=32,
                 use_se_loss=False,
                 add_lateral=False,
                 **kwargs):
        super(EncHead, self).__init__()
        self.use_se_loss = use_se_loss
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.in_channels = [256, 512, 1024, 2048]
        self.channels = 512
        self.num_classes = num_classes
        self.bottleneck = nn.Conv2d(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
            )
        
        if add_lateral:
            self.lateral_convs = nn.ModuleList()
            for in_channels in self.in_channels[:-1]:  # skip the last one
                self.lateral_convs.append(
                    nn.Conv2d(
                        in_channels,
                        self.channels,
                        1,
                        ))
            self.fusion = nn.Conv2d(
                len(self.in_channels) * self.channels,
                self.channels,
                3,
                padding=1,
                )
            
        self.enc_module = EncModule(
            self.channels,
            num_codes=num_codes,
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 2, 3, padding=1)
        )
        
        if self.use_se_loss:
            self.se_layer = nn.Linear(self.channels, self.num_classes)
 
    def forward(self, inputs):
        """Forward function."""
        feat = self.bottleneck(inputs[-1])
        if self.add_lateral:
            laterals = [
                nn.functional.interpolate(input=lateral_conv(inputs[i]),size=feat.shape[2:],
                    mode='bilinear')
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]
            feat = self.fusion(torch.cat([feat, *laterals], 1))
        encode_feat, output = self.enc_module(feat)
        output = nn.functional.interpolate(input = output, scale_factor=8, mode="bilinear")
        output = self.cls_seg(output)
        if self.use_se_loss:
            se_output = self.se_layer(encode_feat)
            return output, se_output
        else:
            return output
 
class ENCNet(nn.Module):
    def __init__(self, num_classes, **_):
        super(ENCNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = ResNet.resnet50()
        self.decoder = EncHead()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x