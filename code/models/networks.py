'''
Author: your name
Date: 2021-07-07 10:32:59
LastEditTime: 2021-11-18 17:00:34
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \code\hdrunetplus\codes\models\networks.py
'''
import imp
from shutil import which
import torch
import logging


###############################################

# brain

import models.modules.newIdea.scale_3_brain_new1 as scale_3_brain_new1 

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == "base_R":
        pass

    elif which_model == "scale_3_brain_new1":
        netG = scale_3_brain_new1.KUNet2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], act_type=opt_net['act_type'])

    

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG

    
    

import models.modules.Discriminator.net1_vgg as net1_vgg
import models.modules.Discriminator.patch_gan as patch_gan
# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = net1_vgg.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == "nf_layer": # Pixel2Pixel
        netD = patch_gan.NLayerDiscriminator(input_nc=opt_net['in_nc'], ndf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = net1_vgg.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF