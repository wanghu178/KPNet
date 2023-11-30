import torch
import torch.nn as nn
import models.modules.wh_utils as wh_util

class KUNet2(nn.Module):
    def __init__(self,in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu'):
        super(KUNet2,self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)

        self.recon_trunk1 = wh_util.mulRDBx4(nf,nf,2)
        self.recon_trunk2 = wh_util.mulRDBx4(nf,nf,2)
        self.recon_trunk3 = wh_util.mulRDBx4(nf,nf,2)


        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)


        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):
        # x[0]: img; x[1]: cond


        fea0 = self.act(self.conv_first(x[0]))

        #fea0 = self.conv_2(fea0)
        #fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1= self.recon_trunk1(fea1)

        fea2 = self.act(self.down_conv2(fea1))
        out = self.recon_trunk2(fea2)
        #out = self.recon_trunk2(out)
        out = out + fea2

        out = self.act(self.up_conv1(out)) + fea1
        out = self.recon_trunk3(out)

        out = self.act(self.up_conv2(out)) + fea0
        

        out = self.conv_last(out)
        out = out
        return out