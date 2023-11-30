import argparse
from operator import index
import os
import utils
import os.path as osp
from yaml import parse
import option
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help="Path of yaml file")
    args = parser.parse_args()
    opt = option.parse(args.opt)

    ref_dir = opt['ref_dir']
    fake_dir = opt['fake_dir']

    psnr = 0 
    idx = 0
    deltaE_ITP  = 0
    ssim_avg = 0
    mse_avg = 0

    for filename in tqdm(os.listdir(ref_dir)):
        image_id = int(filename[:4])
        
        ref_hdr_image = utils.read_img(env=None,path=osp.join(ref_dir, "{:04d}.png".format(image_id)))
        fake_hdr_image = utils.read_img(env=None,path=osp.join(fake_dir, "{:04d}.png".format(image_id)))
        
        temp =  utils.calculate_psnr(ref_hdr_image,fake_hdr_image)
        temp_deltaE_ITP = utils.calculate_hdr_deltaITP(ref_hdr_image,fake_hdr_image)
        temp_ssim = utils.calculate_ssim(ref_hdr_image,fake_hdr_image)
        temp_mse = utils.calculate_mse(ref_hdr_image,fake_hdr_image)
        ssim_avg += temp_ssim
        psnr+=temp
        deltaE_ITP+=temp_deltaE_ITP
        mse_avg += temp_mse

        idx+=1
       # if(idx == 2): break
    print("psnr:",psnr/idx,"deltaE_ITP",deltaE_ITP/idx,"ssim: ",ssim_avg/idx,"mse:",mse_avg/idx)



if __name__ == '__main__':
    main()
    