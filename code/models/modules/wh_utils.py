from selectors import SelectorKey
from turtle import forward
import torch
import torch.nn as nn 
import torch.nn.functional as F
from  models.modules.attention import *
import numpy as np
# 错位变换
class dislocation_change(nn.Module):
    def __init__(self,ch):
        super(dislocation_change,self).__init__()
        self.conv1 = nn.Conv2d(ch,ch,3,1,1)
        self.conv2 = nn.Conv2d(ch,ch,3,1,1)
        self.act = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(ch*3,ch,1,1,0)
    def forward(self,x):
        out1 = self.act(self.conv1(x))+x
        out2 = self.act(self.conv2(out1))
        result = self.fuse(torch.cat((out1,out2,x),dim=1))
        return  result




#1*1激活块
class conv_1x1(nn.Module):
    def __init__(self,ch):
        super(conv_1x1,self).__init__()
        self.conv = nn.Conv2d(ch,ch,1,1,0)
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.act(self.conv(x))
        return out 


#----------------------------------------------------------------------
class mulRDBx2(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDBx2,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*2,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        f_c = torch.cat((rdb1,rdb2),dim=1)
        out = self.conv1x1(f_c)
        return out




#############################################################
#################HDR生成模块##################################
class mulRDBx6(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDBx6,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*6,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        rdb3 = self.rdb(rdb2)
        rdb4 = self.rdb(rdb3)
        rdb5 = self.rdb(rdb4)
        rdb6 = self.rdb(rdb5)
        f_c = torch.cat((rdb1,rdb2,rdb3,rdb4,rdb5,rdb6),dim=1)
        out = self.conv1x1(f_c)
        return out

class mulRDBx4(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDBx4,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*4,growth_rate,1,1,0)
    def forward(self,x):
        rdb1 = self.rdb(x)
        rdb2 = self.rdb(rdb1)
        rdb3 = self.rdb(rdb2)
        rdb4 = self.rdb(rdb3)
        f_c = torch.cat((rdb1,rdb2,rdb3,rdb4),dim=1)
        out = self.conv1x1(f_c)
        return out

    

##---------高低频分离处理------------------
class mulRDB_hl(nn.Module):
    def __init__(self,intput_dim,growth_rate=64,nb_layers=2):
        super(mulRDB_hl,self).__init__()
        self.rdb = RDB(intput_dim,growth_rate,nb_layers)
        self.conv1x1 = conv2d(growth_rate*5,growth_rate,1,1,0,acti='relu')
        
        self.asymmetric_block = asymmetric_block(nf=growth_rate)

    def forward(self,x):
        rdb1 = self.rdb(x)
        
        asymmetric_block1 = self.asymmetric_block(rdb1)
        
        rdb2 = self.rdb(asymmetric_block1)
        
        asymmetric_block2 = self.asymmetric_block(rdb2)

        rdb3 = self.rdb(asymmetric_block2)
        f_c = torch.cat((rdb1,asymmetric_block1,rdb2,asymmetric_block2,rdb3),dim=1)
        out = self.conv1x1(f_c)
        return out

class R_hl(nn.Module):
    def __init__(self,nf):
        super(R_hl,self).__init__()
        self.split_hl = SplitHLfrequence(nf,nf,(3,3))
        self.conv_h = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))
        self.conv_l = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))

        self.up_sampel = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0,acti='relu'))
        
    def forward(self,x):
        h,l = self.split_hl(x)
        x_h = self.conv_h(h) # 处理之后的高频信息
        x_l = self.conv_l(l) # 处理之后的低频信息
        x_middle = x
        result = x_h + self.up_sampel(x_l) + x_middle
        result_refine = self.refine(result)
        return result_refine

# R_hl 高低频信息正则化
class R_hl_normalization(nn.Module):
    def __init__(self,nf):
        super(R_hl_normalization,self).__init__()
        self.split_hl = SplitHLfrequence(nf,nf,(3,3))
        self.conv_h = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))
        self.conv_l = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))

        self.up_sampel = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0,acti='relu'))
        self.batchnorm = nn.BatchNorm2d(nf, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True, device=None, dtype=None)
        
    def forward(self,x):
        h,l = self.split_hl(x)
        h = self.batchnorm(h)
        l = self.batchnorm(l)
        x = self.batchnorm(x)
        x_h = self.conv_h(h) # 处理之后的高频信息
        x_l = self.conv_l(l) # 处理之后的低频信息
        
        # 正则化高低频信息
        
        x_middle = x
        result = x_h + self.up_sampel(x_l) + x_middle
        result_refine = self.refine(result)
        return result_refine


# 对base块进行修正
class base_2(nn.Module):
    def __init__(self,nf):
        super(base_2,self).__init__()
        self.split_hl = SplitHLfrequence(nf,nf,(3,3))
        self.conv_h = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))
        self.conv_l = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))

        self.up_sampel = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.refine = mulRDBx2(nf)
        
        self.fuse_hl = nn.Conv2d(nf*2,nf,1,1,0)
    def forward(self,x):
        h,l = self.split_hl(x)
        x_h = self.conv_h(h) # 处理之后的高频信息
        x_l = self.conv_l(l) # 处理之后的低频信息
        x_middle = x
        # 这样做使得压缩鬼影的恢复用残差
        result = self.fuse_hl(torch.cat((x_h,self.up_sampel(x_l)),dim=1))  + x_middle
        result_refine = self.refine(result)
        return result_refine







#--------------------------------#
#对精炼层进行改进     

class R_hl_asy(nn.Module):
    def __init__(self,nf):
        super(R_hl_asy,self).__init__()
        self.split_hl = SplitHLfrequence(nf,nf,(3,3))
        self.conv_h = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))
        self.conv_l = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),
                                    conv2d(nf,nf,3,1,1))


        self.conv1_1_h = nn.Sequential(nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv1_2_h = nn.Sequential(nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv1_3_h = nn.Sequential(nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))


        self.conv1_1_l = nn.Sequential(nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=(1,3),padding=(0,1),groups=1,bias=False))
        self.conv1_2_l = nn.Sequential(nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=(3,1),padding=(1,0),groups=1,bias=False))
        self.conv1_3_l = nn.Sequential(nn.Conv2d(in_channels=nf,out_channels=nf,kernel_size=(3,3),padding=(1,1),groups=1,bias=False))


        self.up_sampel = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
    def forward(self,x):
        h,l = self.split_hl(x)
        
        h1_1 = self.conv1_1_h(h)
        h1_2 = self.conv1_2_h(h)
        h1_3 = self.conv1_3_h(h)
        h1 = h1_1 + h1_2 + h1_3
        x_h = self.acti(h1)

        l1_1 = self.conv1_1_l(l)
        l1_2 = self.conv1_2_l(l)
        l1_3 = self.conv1_3_l(l)
        l1 = l1_1 + l1_2 + l1_3
        x_l = self.acti(l1)

        
        x_middle = x

        result = x_h + self.up_sampel(x_l) + x_middle
        result_refine = self.refine(result)

        return result_refine


class brain_1(nn.Module):
    def __init__(self,nf):
        super(brain_1,self).__init__()
        self.conv1 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))

        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
    def forward(self,one,two,three):
        
        compution1_1 = one + two
        compution1_2 = one + three
        compution1_3 = two + three

        compution1_1_act = self.acti(compution1_1)
        compution1_2_act = self.acti(compution1_2)
        compution1_3_act = self.acti(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)

        compution3 = compution2_1 + compution2_2 + compution2_3
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result

    def __init__(self,nf):
        super(brain_1,self).__init__()
        self.one_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.two_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.three_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))

        self.compution_1x1 = nn.Sequential(conv2d(nf*3,nf,1,1,0))
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)

        # 为融合设置权重 独立参数
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # initialization
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        compution1_1_act = self.one_1x1(compution1_1)
        compution1_2_act = self.two_1x1(compution1_2)
        compution1_3_act = self.three_1x1(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)*self.w1
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)*self.w3

        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.compution_1x1(torch.cat((compution2_2,compution2_1,compution2_3),dim=1))
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result

class brain_1_cat(nn.Module):
    def __init__(self,nf):
        super(brain_1_cat,self).__init__()
        self.one_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.two_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.three_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
    
    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        compution1_1_act = self.one_1x1(compution1_1)
        compution1_2_act = self.two_1x1(compution1_2)
        compution1_3_act = self.three_1x1(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)

        compution3 = compution2_1 + compution2_2 + compution2_3
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result

class brain_1_catPlus(nn.Module):
    def __init__(self,nf):
        super(brain_1_catPlus,self).__init__()
        self.one_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.two_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.three_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
    
    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        compution1_1_act = self.one_1x1(compution1_1)
        compution1_2_act = self.two_1x1(compution1_2)
        compution1_3_act = self.three_1x1(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)

        compution3 = compution2_1 + compution2_2 + compution2_3
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result

class brain_1_catPlusShare(nn.Module):
    def __init__(self,nf):
        super(brain_1_catPlusShare,self).__init__()
        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
    
    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)

        compution3 = compution2_1 + compution2_2 + compution2_3
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result

# 对三个需要融合的模块施加注意力机制

class brain_six_XY(nn.Module):
    def __init__(self,nf):
        super(brain_six_XY,self).__init__()
        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = Att(nf)
    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)
        
        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2) + self.nm_att(compution2_3)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result
# 对注意力修正
class brain_new1(nn.Module):
    def __init__(self,nf):
        super(brain_new1,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)
        
        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2) + self.nm_att(compution2_3)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result

class brain_new2(nn.Module):
    def __init__(self,nf):
        super(brain_new2,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one,two,three):
        '''
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)

        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)
        '''
        compution2_1 = self.conv1(one)
        compution2_2 = self.conv2(two)
        compution2_3 = self.conv3(three)

        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2) + self.nm_att(compution2_3)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)

        return result

class brain_new3(nn.Module):
    def __init__(self,nf):
        super(brain_new3,self).__init__()

    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))
        #self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1 = 0.3
        self.w2 = 0.4
        self.w3 = 0.3
        self.acti = nn.ReLU(inplace=True)

    def forward(self,one,two,three):
        
      
        compution3 = one*self.w1 + two*self.w2 + three*self.w3
        compution3_act = self.acti(compution3)
        result = self.refine(compution3_act)
        return result

class brain_newAllCat(nn.Module):
    def __init__(self,nf):
        super(brain_newAllCat,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf*3,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one,two,three):
        
        compution1 = torch.cat((one,two,three),dim=1)
        compution2 = self.conv_1x1Share(compution1)
        
        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2) 
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result


class brain_new4(nn.Module):
    def __init__(self,nf):
        super(brain_new4,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        #self.nm_att = CoordAtt(nf,nf)
        self.nm_att1 = cooperate_attention(nf)
        self.nm_att2 = cooperate_attention(nf)
        self.nm_att3 = cooperate_attention(nf)
        self.alpha = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        self.belta = nn.Sequential(nn.Conv2d(nf,nf,1,1,0),nn.Conv2d(nf,nf,1,1,0))
        

    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)

        compution2_1 = self.conv1(compution1_1_act) 
        compution2_2 = self.conv2(compution1_2_act) 
        compution2_3 = self.conv3(compution1_3_act) 
        

        compution3 = compution2_1*self.nm_att1(compution1_1) + compution2_2*self.nm_att2(compution1_2) + compution2_3*self.nm_att3(compution1_3)
       # compution3 = self.nm_att1(compution2_1) + self.nm_att2(compution2_2) + self.nm_att3(compution2_3)
        
        compution3_act = self.acti(compution3)
        result = self.refine(compution3_act)
        return result




class brain_kib4(nn.Module):
    def __init__(self,nf):
        super(brain_kib4,self).__init__()
        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))

        self.convAD = nn.Sequential(res_block(nf,acti='relu'))
        self.convBD = nn.Sequential(res_block(nf,acti='relu'))
        self.convCD = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = Att(nf)
    def forward(self,one,two,three,four):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)

        compution1_4 = torch.cat((one,four),dim=1)
        compution1_5 = torch.cat((two,four),dim=1)
        compution1_6 = torch.cat((three,four),dim=1)
        
        
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)

        compution1_4_act = self.conv_1x1Share(compution1_4)
        compution1_5_act = self.conv_1x1Share(compution1_5)
        compution1_6_act = self.conv_1x1Share(compution1_6)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)

        compution2_4 = self.convAD(compution1_4_act)
        compution2_5 = self.convBD(compution1_5_act)
        compution2_6 = self.convCD(compution1_6_act)
        
        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2) + self.nm_att(compution2_3) +\
                    self.nm_att(compution2_4) +  self.nm_att(compution2_5) + self.nm_att(compution2_6)
                    
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result


class brain_kib2(nn.Module):
    def __init__(self,nf):
        super(brain_kib2,self).__init__()
        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one,two):
        
        
        compution1_1 = torch.cat((one,two),dim=1)
        
        compution1_1_act = self.conv_1x1Share(compution1_1)

        compution2_1 = self.conv1(compution1_1_act)
        
        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result


class brain_new2(nn.Module):
    def __init__(self,nf):
        super(brain_new2,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one,two,three):
        '''
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)

        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)
        '''
        compution2_1 = self.conv1(one)
        compution2_2 = self.conv2(two)
        compution2_3 = self.conv3(three)

        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2) + self.nm_att(compution2_3)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)

        return result
    

class brain_new3_kib2(nn.Module):
    def __init__(self,nf):
        super(brain_new3_kib2,self).__init__()

    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))
        #self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1 = 0.5
        self.w2 = 0.5
        self.acti = nn.ReLU(inplace=True)

    def forward(self,one,two):
        
        compution3 = one*self.w1 + two*self.w2 
        compution3_act = self.acti(compution3)
        result = self.refine(compution3_act)
        
        return result

class brain_3_cat(nn.Module):
    def __init__(self,nf):
        super(brain_3_cat,self).__init__()
        self.one_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.two_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))
        self.three_1x1 = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(conv2d(nf,nf,3,1,1),nn.ReLU(inplace=True))

        # 为融合设置权重 独立参数
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        # initialization
        self.w1.data.fill_(0.3)
        self.w2.data.fill_(0.4)
        self.w3.data.fill_(0.3)


        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)

    def forward(self,one,two,three):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        compution1_1_act = self.one_1x1(compution1_1)
        compution1_2_act = self.two_1x1(compution1_2)
        compution1_3_act = self.three_1x1(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)*self.w1
        compution2_2 = self.conv2(compution1_2_act)*self.w2
        compution2_3 = self.conv3(compution1_3_act)*self.w3

        compution3 = compution2_1 + compution2_2 + compution2_3
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result
# seven 全连接不行 ，继续修正将one,two,three均加上注意力依旧不行 继续修正。
class brain_seven_XY(nn.Module):
    def __init__(self,nf):
        super(brain_seven_XY,self).__init__()
        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(res_block(nf,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = Att(nf)
    def forward(self,one,two,three):
        # one,two,three = self.nm_att(one),self.nm_att(two),self.nm_att(three)
       
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,three),dim=1)
        compution1_3 = torch.cat((two,three),dim=1)
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)
        compution1_3_act = self.conv_1x1Share(compution1_3)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        compution2_3 = self.conv3(compution1_3_act)
        
        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2) + self.nm_att(compution2_3)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        return result
    
    
# ablation


class brain_new1_inter2_1(nn.Module):
    def __init__(self,nf):
        super(brain_new1_inter2_1,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one,two):
        
        compution1_1 = torch.cat((one,two),dim=1)
        compution1_2 = torch.cat((one,two),dim=1)
        
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        
        #compution3 = compution2_1 + compution2_2 + compution2_3
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result



class brain_new1_inter1_1(nn.Module):
    def __init__(self,nf):
        super(brain_new1_inter1_1,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one):
    
        compution1_1 = one
        
        compution1_1_act = self.conv_1x1Share(compution1_1)

        compution2_1 = self.conv1(compution1_1_act)
        
        compution3 = self.nm_att(compution2_1)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result






class brain_new1_inter2_2(nn.Module):
    def __init__(self,nf):
        super(brain_new1_inter2_2,self).__init__()

        self.conv_1x1Share = nn.Sequential(conv2d(nf*2,nf,1,1,0),nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv2 = nn.Sequential(res_block(nf,acti='relu'))
        self.conv3 = nn.Sequential(res_block(nf,acti='relu'))
    
        self.refine = nn.Sequential(conv2d(nf,nf,3,1,1,acti='relu'),conv2d(nf,nf,1,1,0))

        self.acti = nn.ReLU(inplace=True)
        
        # 注意力机制
        self.nm_att = CoordAtt(nf,nf)
    def forward(self,one,two):
        
        compution1_1 = torch.cat((one,one),dim=1)
        compution1_2 = torch.cat((one,two),dim=1)
        
        compution1_1_act = self.conv_1x1Share(compution1_1)
        compution1_2_act = self.conv_1x1Share(compution1_2)

        compution2_1 = self.conv1(compution1_1_act)
        compution2_2 = self.conv2(compution1_2_act)
        
        compution3 = self.nm_att(compution2_1) + self.nm_att(compution2_2)
        compution3_act = self.acti(compution3)

        result = self.refine(compution3_act)
        
        return result

#############################################################
#----------------------高低频信息处理模块----------------------

class handle_hl_info(nn.Module):
    def __init__(self,in_channel,nf):
        super(handle_hl_info,self).__init__()
        self.att = base_channel_attention(in_channel,reduction=16)
        self.conv2d_h = conv2d(in_channel,nf,3,1,1,acti='relu')
        self.conv2d_l = conv2d(in_channel,nf,3,1,1,acti='relu')
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv_1x1 = conv_1x1(nf)
    def forward(self,h,l):
        h_result = self.conv2d_h(h)
        l_result = self.upsample(self.conv2d_l(l))
        x = h_result+l_result
        x = self.conv_1x1(x)
        return x

class handle_hl_info_rem(nn.Module):
    def __init__(self,in_channel,nf):
        super(handle_hl_info_rem,self).__init__()
        self.att = base_channel_attention(in_channel,reduction=16)
        self.conv2d_h = conv2d(in_channel,nf,3,1,1,acti='relu')
        self.conv2d_h_rem = conv2d(in_channel,nf,3,1,1,acti='relu')

        self.conv2d_l = conv2d(in_channel,nf,3,1,1,acti='relu')
        self.conv2d_l_rem = conv2d(in_channel,nf,3,1,1,acti='relu')
        
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv_1x1 = conv2d(nf*2,nf,1,1,0)
    def forward(self,h,l,h_rem,l_rem):
        h_result = self.conv2d_h(h)
        h_result = torch.cat((h_result,h_rem),dim=1)
        
        l_result = self.upsample(self.conv2d_l(l))
        l_result = torch.cat((l_result,l_rem),dim=1)
        x = h_result+l_result
        x = self.conv_1x1(x)
        return x


#################################################
#------------------XY改进------------------------
# 并行
class XY2(nn.Module):
    def __init__(self,nf):
        super(XY2,self).__init__()
        self.conv_up = conv2d(nf,nf,1,1,0,acti='relu')
        self.conv_down = conv2d(nf,nf,1,1,0,acti='relu')
        self.out_fuse = conv2d(nf*2,nf,1,1,0)
    def forward(self,x):
        x_up = self.conv_up(x)
        x_down = self.conv_down(x)
        out = torch.cat((x_up,x_down),dim=1)
        out = self.out_fuse(out)
        return out
 
class XY3(nn.Module):
    def __init__(self,nf):
        super(XY3,self).__init__()
        self.conv_up1 = conv2d(nf,nf,1,1,0,acti='relu')
        self.conv_down1 = conv2d(nf,nf,1,1,0,acti='relu')
        self.out_fuse = conv2d(nf*2,nf,1,1,0)

        self.conv_up2 = conv2d(nf,nf,1,1,0)
        self.conv_down2 = conv2d(nf,nf,1,1,0)

    def forward(self,out):
        x = out
        x_up = self.conv_up1(x)
        x_down = self.conv_down1(x)
        fuse = torch.cat((x_up,x_down),dim=1)
        fuse = self.out_fuse(fuse)

        alpha = self.conv_up2(fuse)
        belta = self.conv_down2(fuse)
        return alpha,belta
# 注意力机制
class Att(nn.Module):
    def __init__(self, channels):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)

        return x_out1  
#######################################
#############基础模块###################
class BasicBLock(nn.Module):
    def __init__(self,intput_dim,output_dim):
        super(BasicBLock,self).__init__()
        self.conv = conv2d(intput_dim,output_dim,3,1,1,acti='relu')
    def forward(self,x):
        out = self.conv(x)
        return x+out 

class RDB(nn.Module):
    def __init__(self,input_dim,growth_rate,nb_layers):
        super(RDB,self).__init__()
        self.layer = self._makekayer(nb_layers,input_dim,growth_rate)
        self.conv1x1 = conv2d(input_dim,growth_rate,1,1,0) 
    def _makekayer(self,nb_layers,intput_dim,growth_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBLock(intput_dim,growth_rate))
        return nn.Sequential(*layers)
    def forward(self,x):
        out = self.layer(x)
        out = self.conv1x1(out)
        return out+x




#定义残差块RB模块
class res_block(nn.Module):
    def __init__(self,c,k_s=3,s=1,p=1,
                d=1,g=1,b=True,acti=None,norm=None):
        super(res_block,self).__init__()
        self.conv1 = conv2d(c,c,k_s,s,p,acti=acti,norm=norm)
        self.conv2 = conv2d(c,c,k_s,s,p,acti=acti,norm=norm)
    def forward(self,x):
        n = self.conv1(x)
        n = self.conv2(n)
        out = x+n
        return out

class conv2d(nn.Module):
    def __init__(self,in_c,out_c,k_s,s,p,
                d=1,g=1,b=True,acti=None,norm=None):
        super(conv2d,self).__init__()
        if acti == 'relu':
            self.acti = nn.ReLU()
        elif acti == 'leak':
            self.acti = nn.LeakyReLU(0.1,inplace=True)
        elif acti == 'selu':
            self.acti = nn.SELU()
        elif acti == 'tanh':
            self.acti = nn.Tanh()
        elif acti == 'sigmod':
            self.acti = nn.Sigmoid()
        elif acti == None:
            self.acti = None
        else:
            raise RuntimeError("no activation function {}".format(acti))
        
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_c)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_c)
        elif norm == None:
            self.norm = None
        else:
            raise RuntimeError("no norm layer:{}".format(norm))
        self.conv = nn.Conv2d(in_c,out_c,k_s,s,p,d,g,b)
    def forward(self,x):
        out = self.conv(x)
        if self.norm != None:
            out = self.norm(out)
        if self.acti != None:
            out = self.acti(out)
        return out




#----------------------------------------------------------
#-----------------高低频分离--------------------------------
class  SplitHLfrequence(nn.Module):
# split the frequence information of high and low
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(SplitHLfrequence, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels,  out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels ,
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class Interactive_hl(nn.Module):
#Interactivae the frequence infomation of high and low 
    def __init__(self,in_channel,nf):
        super(Interactive_hl,self).__init__()

        self.conv2d_h = conv2d(in_channel,nf,3,1,1,acti='relu')
        self.conv2d_l = conv2d(in_channel,nf,3,1,1,acti='relu')
        
        self.conv2d_h_1 = nn.Sequential(conv2d(in_channel,nf,3,1,1,acti='relu'))
        self.conv2d_l_1 = nn.Sequential(conv2d(in_channel,nf,3,1,1,acti='relu'))

        #self.upsample = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.refine = nn.Sequential(conv2d(in_channel,nf,3,1,1))
        
        self.acti = nn.ReLU(inplace=True)
    def forward(self,h,l):
        h_1 = self.conv2d_h_1(h)
        l_1 = self.conv2d_l_1(l)

        l_up  = self.upsample(l_1)

        h_2 = self.conv2d_h(h_1+l_up)
        l_2 = self.conv2d_l(h_2)

        fuse = h_2+l_2

        result = self.refine(fuse)
        return result

class asymmetric_block(nn.Module):
    def __init__(self,nf):
        super(asymmetric_block,self).__init__()
        self.split_hl = SplitHLfrequence(nf,nf,(3,3))
        self.fuse_hl = Interactive_hl(nf,nf)
    def forward(self,x):
        h,l = self.split_hl(x) # split the frequence information of high and low  from x
        output = self.fuse_hl(h,l)
        return output

        





#------------------------------------------------------
#--------------octconv--------------------------------

##########构建一个用于知道三个特征汇合的注意力机制
class cooperate_attention(nn.Module):
    def __init__(self,nf):
        super(cooperate_attention,self).__init__()
        #self.fuse_cat = nn.Conv2d(nf*2,nf,1,1,0)
        self.conv_deal = nn.Sequential(
                                    nn.Conv2d(nf*2,nf,3,1,1),
                                    nn.LeakyReLU(0.01),
                                    nn.AvgPool2d(3,2,1),
                                    nn.Conv2d(nf,nf,3,1,1),
                                    nn.LeakyReLU(0.01),
                                    nn.AdaptiveAvgPool2d(1))
        self.acti = nn.Softmax(dim=1)
    def forward(self,x):
        x_fea = self.conv_deal(x)
        x_att = self.acti(x_fea)
        return x_att

###############新的大脑块使用尺度见聚合###############
class cpm_key_component1(nn.Module):
    def __init__(self,nf):
        super(cpm_key_component1,self).__init__()
        self.up_conv = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.cat_fuse = nn.Sequential(
            nn.Conv2d(nf*2,nf,1,1,0),
            nn.LeakyReLU(0.01),
            nn.Conv2d(nf,nf,3,2,1),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d(1)
        ) 
        self.linear_adjust_x = nn.Conv2d(nf,nf,1,1,0,bias=True)
        self.linear_adjust_y = nn.Conv2d(nf,nf,1,1,0,bias=True)
        self.acti = nn.Softmax(dim=1)

        self.refine = nn.Sequential(mulRDBx2(nf))
class cpm_key_component2(nn.Module):
    def __init__(self,nf):
        super(cpm_key_component2,self).__init__()
        self.up_conv = nn.Sequential(nn.Conv2d(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.enhance_y = nn.Conv2d(nf,nf,3,1,1)
        self.cat_fuse = nn.Sequential(
            nn.Conv2d(nf*2,nf,1,1,0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf,nf,3,1,1),
            nn.ReLU(inplace=True),
        ) 

        self.refine = nn.Sequential(res_block(nf),nn.Conv2d(nf,nf,3,1,1))
    def forward(self,x,y):
        y_up = self.up_conv(y)
        y_up = self.enhance_y(y_up)
        fuse1 = self.cat_fuse(torch.cat((x,y_up),dim=1)) 
        result = fuse1 + x

        result_fine = self.refine(result)
        return result_fine

class cpm_1(nn.Module):
    def __init__(self,nf):
        super(cpm_1,self).__init__()
        #self.component1 = cpm_key_component2(nf)
        #self.component2 = cpm_key_component2(nf)
        #self.component3 = cpm_key_component2(nf)

        self.component = cpm_key_component2(nf)

    def forward(self,x,y,z):
        # x,y,z: 尺度逐渐减小
        aggregation_1 = self.component(x,y)
        aggregation_2 = self.component(y,z)
        aggregation_3 = self.component(aggregation_1,aggregation_2)
        return aggregation_3
