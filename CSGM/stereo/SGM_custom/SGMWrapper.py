"""
利用半全局匹配（SGM）算法计算视差图。
"""

import numpy as np
import cv2

def SGMWrapper(im_left, im_right, params):
    block_size = params.get('block_size', 5)
    disp_range = params.get('disparity_range', (-16, 16))
    min_disp = disp_range[0]
    max_disp = disp_range[1]
    num_disp = max_disp - min_disp + 1
    
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=block_size)
    disparity = stereo.compute(im_left, im_right).astype(np.float32) / 16.0
    # C 为代价矩阵，此处简单返回零矩阵（实际算法中需计算成本）
    C = np.zeros_like(disparity)
    return disparity, C