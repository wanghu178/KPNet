#divide from middle
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from progress_bar import ProgressBar
def worker(path,save_folder,compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
    h,w,c = img.shape
    height = h // 2
    width = w // 2
    seq = 1
    for i in range(2):
        for j in range(2):
            img_roi = img[(i*height):((i+1)*height),(j*width):((j+1)*width),:]
            name = str(seq)+img_name
            image_save_path = os.path.join(save_folder,name)
            cv2.imwrite(image_save_path,img_roi,[cv2.IMWRITE_PNG_COMPRESSION, compression_level])
            seq += 1
    return 'Processing {:s} ...'.format(img_name)

def main():
    input_folder = r'E:\hdrtv\test_set\test_hdr' 
    save_folder = r'E:\hdrtv\test_set\test_hdr_sub'
    #input_folder = r'D:\dataset\compress\qp42\test_sdr_42_png' 
    #save_folder = r'D:\dataset\compress\qp42\test_sdr_42_png_sub'
    
    compression_level = 0
    n_thread = 2

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
            args=(path, save_folder, compression_level),
            callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')



if __name__ == '__main__':
    main()
