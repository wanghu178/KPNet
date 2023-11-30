from unittest import result
import torch
import torch.nn as nn
import models.modules.wh_utils as wh_util
'''
sixAtt: 引导信息用注意力计算权重 5.17
'''
class KUNet2(nn.Module):
    def __init__(self,in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(KUNet2,self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)


        # self.recon_trunk1 = wh_util.R_hl(nf)
        # self.recon_trunk2 = wh_util.R_hl(nf)
        # self.recon_trunk3 = wh_util.R_hl(nf)

        self.recon_trunk = wh_util.R_hl(nf)

        # 错位变换
        self.dislocation_change1 =  wh_util.dislocation_change(nf)

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

      

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.alpha = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        self.belta = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        
        
    def forward(self, x):
            

        fea0 = self.act(self.conv_first(x[0]))

        #fea1 = self.act(self.down_conv1(fea0))
        fea_kib1 = self.dislocation_change1(fea0)
        KIB_1 = self.recon_trunk(fea_kib1)*self.alpha(fea_kib1) + self.belta(fea_kib1)
        
        result = self.conv_last(KIB_1)
        return result