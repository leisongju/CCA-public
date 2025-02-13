"""
基于 MATLAB calc_int.m，实现低分辨率视差图的计算。
"""

import cv2
import numpy as np
import time
from calc_int_stereo_mid_image import disparCostAltConf

def main():
    imgset = 'training'
    imgsize = 'Q'
    path_save = r'C:\Users\local_admin\Documents\GitHub\C-SGM\CSGM\stereo\int_disparity_calc'
    
    im_num = 1
    # 示例加载图像，实际中请调用恰当的数据加载函数
    im_L = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
    im_R = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)
    im_guide = im_L
    params = {'cost': 'BT', 'gaussKerSigma': 5, 'dispRange': (-10, 10)}
    
    cost_neig = {}
    conf_score = {}
    dispar_int_val = {}
    cost_neig_R = {}
    conf_score_R = {}
    dispar_int_val_R = {}
    
    start = time.time()
    cost_neig[im_num], conf_score[im_num], dispar_int_val[im_num] = disparCostAltConf(im_L, im_R, params)
    params['dispar_vals'] = -np.flip(np.arange(params['dispRange'][0], params['dispRange'][1]+1))
    cost_neig_R[im_num], conf_score_R[im_num], dispar_int_val_R[im_num] = disparCostAltConf(im_R, im_L, params)
    end = time.time()
    
    print("计算完成，耗时：", end - start)
    curr_save = f"{path_save}\\tgt{im_num}_res{imgsize}.npz"
    np.savez(curr_save, im_L=im_L, im_R=im_R, im_guide=im_guide,
             cost_neig=cost_neig, conf_score=conf_score, dispar_int_val=dispar_int_val,
             cost_neig_R=cost_neig_R, conf_score_R=conf_score_R, dispar_int_val_R=dispar_int_val_R)

if __name__ == '__main__':
    main()