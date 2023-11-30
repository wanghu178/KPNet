import numpy as np
import math
import cv2
import os
import colour
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32) # np.float64
    img2 = img2.astype(np.float32) # np.float64
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))


def read_img(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        if os.path.splitext(path)[1] == '.npy':
            img = np.load(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        pass
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
       
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
      
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img



# \delta_{ITP} 
def calculate_hdr_deltaITP(img1, img2):
    img1 = img1[:, :, [2, 1, 0]]
    img2 = img2[:, :, [2, 1, 0]]
    img1 = colour.models.eotf_ST2084(img1)
    img2 = colour.models.eotf_ST2084(img2)
    img1_ictcp = colour.RGB_to_ICTCP(img1)
    img2_ictcp = colour.RGB_to_ICTCP(img2)
    delta_ITP = 720 * np.sqrt((img1_ictcp[:,:,0] - img2_ictcp[:,:,0]) ** 2
                            + 0.25 * ((img1_ictcp[:,:,1] - img2_ictcp[:,:,1]) ** 2)
                            + (img1_ictcp[:,:,2] - img2_ictcp[:,:,2]) ** 2)
    return np.mean(delta_ITP)

def calculate_ssim(img1,img2):
    img1 = img1[:, :, [2, 1, 0]]
    img2 = img2[:, :, [2, 1, 0]]
    sim = ssim(img1,img2,multichannel=True)
    return sim

def calculate_mse(img1,img2):
    
    cal_mse = mse(img1,img2)
    return cal_mse



##############################################################################
#----------------------------NTIRE计算--------------------------------
import numpy as np
def read_npy(path):
    return np.load(path)

def read_imgdata(path, ratio=255.0):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED) / ratio

import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
def mu_tonemap(hdr_image, mu=5000):
    """ This function computes the mu-law tonemapped image of a given input linear HDR image.
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        mu (float): Parameter controlling the compression performed during tone mapping.
    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.
    """
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value and then computes
    the mu-law tonemapped image.
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
        mu (float): Parameter controlling the compression performed during tone mapping.
    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.
    """
    return mu_tonemap(hdr_image/norm_value, mu)

def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value, afterwards bounds the
    HDR image values by applying a tanh function and afterwards computes the mu-law tonemapped image.
        the mu-law tonemapped image.
        Args:
            hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
            norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
            mu (float): Parameter controlling the compression performed during tone mapping.
        Returns:
            np.ndarray (): Returns the mu-law tonemapped image.
        """
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)

def psnr_tanh_norm_mu_tonemap(hdr_nonlinear_ref, hdr_nonlinear_res, percentile=99, gamma=2.24):
    """ This function computes Peak Signal to Noise Ratio (PSNR) between the mu-law computed images from two non-linear
    HDR images.
            Args:
                hdr_nonlinear_ref (np.ndarray): HDR Reference Image after gamma correction, used for the percentile norm
                hdr_nonlinear_res (np.ndarray: HDR Estimated Image after gamma correction
                percentile (float): Percentile to to use for normalization
                gamma (float): Value used to linearized the non-linear images
            Returns:
                np.ndarray (): Returns the mean mu-law PSNR value for the complete image.
            """
    hdr_linear_ref = hdr_nonlinear_ref**gamma
    hdr_linear_res = hdr_nonlinear_res**gamma
    norm_perc = np.percentile(hdr_linear_ref, percentile)
    return psnr(tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc), tanh_norm_mu_tonemap(hdr_linear_res, norm_perc))

def psnr(im0, im1):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.
        """
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))

def normalized_psnr(im0, im1, norm):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images that are normalized by the
    specified norm value.
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
            norm (float) : Normalization value for both images.
        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.
        """
    return psnr(im0/norm, im1/norm)

def calculate_tonemapped_psnr(res, ref, percentile=99, gamma=2.24):
    res = res ** gamma
    ref = ref ** gamma
    norm_perc = np.percentile(ref, percentile)
    tonemapped_psnr = -10*np.log10(np.mean(np.power(tanh_norm_mu_tonemap(ref, norm_perc) - tanh_norm_mu_tonemap(res, norm_perc), 2)))
    return tonemapped_psnr

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))

def normalized_ssim(im0, im1, norm):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images that are normalized by the
    specified norm value.
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
            norm (float) : Normalization value for both images.
        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.
        """
    return ssim(im0/norm, im1/norm,multichannel=True)

def mse_hdr(img1,img2):
    """
    This function computea the mse between two images 
    """
    return np.mean((img1-img2)**2)


