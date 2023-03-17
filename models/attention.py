import torch 
import torch.nn as nn
from torch.nn import functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Attention2d(nn.Module):

    def __init__(self, channels, bn_mom=0.99, bn_eps=0.001, reduction=36):
        super().__init__()
        self._channels = channels
        self._bn_mom = bn_mom
        self._bn_eps = bn_eps

        depth_chann = channels * 2
        self._depthwise_conv = nn.Conv1d(
            in_channels=depth_chann, out_channels=depth_chann, groups=depth_chann,
            kernel_size=3, bias=False, padding=1
        )
        self._bn1 = nn.BatchNorm1d(num_features=depth_chann, momentum=self._bn_mom, eps=self._bn_eps)

        se_chann = max(1, depth_chann//reduction)
        self._se_reduce = nn.Conv1d(in_channels=depth_chann, out_channels=se_chann, kernel_size=1, bias=True)

        self._project_conv = nn.Conv1d(
            in_channels=se_chann, out_channels=depth_chann,
            kernel_size=1, bias=True
        )

        self._activate = h_swish()

    def forward(self, inputs):
        
        _, _, _h, _w = inputs.shape
        min_dim = min(_h, _w)

        h = inputs
        h = F.adaptive_avg_pool2d(h, (min_dim, 1)).squeeze(dim=-1)
        w = inputs.transpose(2, 3)
        w = F.adaptive_avg_pool2d(w, (min_dim, 1)).squeeze(dim=-1)

        x = torch.cat((h, w), dim=1)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._activate(x)

        x = self._se_reduce(x)
        x = self._activate(x)

        x = self._project_conv(x).sigmoid()

        h, w = torch.split(x, self._channels, dim=1)
        h = h.unsqueeze(dim=-1)
        w = w.unsqueeze(dim=-2)
        mask = F.interpolate((h * w), size=(_h, _w))
        out = inputs * mask

        return out