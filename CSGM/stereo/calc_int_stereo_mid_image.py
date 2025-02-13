"""
计算中间分辨率下视差图的成本和置信度。
"""

import cv2
import numpy as np
import time

def disparCostAltConf(im_L, im_R, params, im_guide=None):
    # 示例实现：利用块匹配计算成本，实际应依据论文实现
    window_size = int(params.get('gaussKerSigma', 5))
    h, w = im_L.shape
    num_disp = 10
    cost_neig = np.zeros((h, w, 3))
    conf_score = np.zeros((h, w))
    dispar_int_val = np.zeros((h, w))
    return cost_neig, conf_score, dispar_int_val

def main():
    imgset = 'training'
    imgsize = 'Q'
    path_save = r'C:\Users\local_admin\CSGM\CSGM_Monin\stereo\int_disparity_calc'
    
    # 简单示例，加载左右图及引导图（需自己实现加载函数）
    im_L = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
    im_R = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)
    im_guide = im_L
    params = {'cost': 'BT', 'gaussKerSigma': 5, 'dispRange': (-10, 10)}
    
    start = time.time()
    cost_neig, conf_score, dispar_int_val = disparCostAltConf(im_L, im_R, params, im_guide)
    params['dispar_vals'] = -np.flip(np.arange(params['dispRange'][0], params['dispRange'][1]+1))
    cost_neig_R, conf_score_R, dispar_int_val_R = disparCostAltConf(im_R, im_L, params, im_guide)
    end = time.time()
    
    print("计算完成，耗时：", end - start)
    curr_save = f"{path_save}\\tgt1_res{imgsize}.npz"
    np.savez(curr_save, im_L=im_L, im_R=im_R, im_guide=im_guide,
             cost_neig=cost_neig, conf_score=conf_score, dispar_int_val=dispar_int_val,
             cost_neig_R=cost_neig_R, conf_score_R=conf_score_R, dispar_int_val_R=dispar_int_val_R)

if __name__ == '__main__':
    main()