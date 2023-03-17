# https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/networks/CC.py
# CCNet: Criss-Cross Attention for Semantic Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W, device):
    inf = torch.tensor(float("inf")).to(device)
    return -torch.diag(inf.repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CCAttention(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(CCAttention, self).__init__()
        ### 匹配输出维度
        self.match_conv = nn.Conv2d(in_dim, out_dim, 1)

        self.query_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        device = x.device
        m_batchsize, _, height, width = x.size()
        x = self.match_conv(x)
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width, device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x
