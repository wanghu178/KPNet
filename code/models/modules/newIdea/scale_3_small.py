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

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv3 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_fea1 = nn.Conv2d(nf, nf, 3, 2, 1)

        # self.recon_trunk1 = wh_util.R_hl(nf)
        # self.recon_trunk2 = wh_util.R_hl(nf)
        # self.recon_trunk3 = wh_util.R_hl(nf)

        self.recon_trunk = wh_util.R_hl(nf)

        # 错位变换
        self.dislocation_change1 =  wh_util.dislocation_change(nf)
        self.dislocation_change2 =  wh_util.dislocation_change(nf)
        self.dislocation_change3 =  wh_util.dislocation_change(nf)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        

        self.up_KIB2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.up_KIB3 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        #self.up_KIB3_1 = torch.nn.Upsample(scale_factor=4, mode='nearest')

        #self.up_KIB3_2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.brain = wh_util.brain_six_XY(nf)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.alpha = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        self.belta = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        
        
    def forward(self, x):
            

        fea0 = self.act(self.conv_first(x[0]))

        fea1 = self.act(self.down_conv1(fea0))
        fea_kib1 = self.dislocation_change1(fea1)
        KIB_1 = self.recon_trunk(fea_kib1)*self.alpha(fea_kib1) + self.belta(fea_kib1)
        
        fea1_down = self.act(self.down_conv2(fea1))
        fea_kib2 = self.dislocation_change2(fea1_down)
        KIB_2 = self.recon_trunk(fea_kib2)*self.alpha(fea_kib2) + self.belta(fea_kib2)
        
        # 持续进行
        fea_kib3 = self.dislocation_change3(fea1_down)
        KIB_3 = self.recon_trunk(fea_kib3)*self.alpha(fea_kib3) + self.belta(fea_kib3)

        brain_process = self.brain(KIB_1,self.up_KIB2(KIB_2),self.up_KIB3(KIB_3))
        result = self.act(self.up_conv2(brain_process)) 
        
    
        result = self.conv_last(result)
        return result