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
        self.down_fea1 = nn.Conv2d(nf, nf, 3, 2, 1)

        # self.recon_trunk1 = wh_util.R_hl(nf)
        # self.recon_trunk2 = wh_util.R_hl(nf)
        # self.recon_trunk3 = wh_util.R_hl(nf)

        self.recon_trunk = wh_util.R_hl(nf)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.up_KIB2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.brain = wh_util.brain_six_XY(nf)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.alpha = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        self.belta = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        
        self.nam_att = wh_util.Att(nf)
        
        # 引导使用cat代替add
        self.cat1_fuse = nn.Conv2d(2*nf,nf,1,1,0)
        self.cat2_fuse = nn.Conv2d(2*nf,nf,1,1,0)
        self.cat3_fuse = nn.Conv2d(2*nf,nf,1,1,0)

    def forward(self, x):


        fea0 = self.act(self.conv_first(x[0]))

        fea1 = self.act(self.down_conv1(fea0))
        KIB_1 = self.recon_trunk(fea1)*self.alpha(fea1) + self.belta(fea1)
        
        KIB_1_down,fea1_down = self.act(self.down_conv2(KIB_1)),self.act(self.down_conv2(fea1))
        
        #fea2 =self.nam_att(KIB_1_down) + fea1_down
        fea2 = self.cat1_fuse(torch.cat((self.nam_att(KIB_1_down),fea1_down),dim=1))
        KIB_2 = self.recon_trunk(fea2)*self.alpha(fea2) + self.belta(fea2)
        

        out = self.act(self.up_conv1(KIB_2))
        out = self.nam_att(out) 

        guide_fuse = self.cat2_fuse(torch.cat((out,fea1),dim=1))
        KIB_3 = self.recon_trunk(guide_fuse)*self.alpha(guide_fuse) + self.belta(guide_fuse)
        
        brain_process = self.brain(KIB_1,self.up_KIB2(KIB_2),KIB_3)
        result = self.act(self.up_conv2(brain_process)) 
        
    
        result = self.conv_last(result)
        return result