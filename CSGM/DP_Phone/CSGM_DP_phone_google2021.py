"""
针对 google2021 手机数据集的 CSGM 算法示例脚本。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))




import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.loadPhone import loadPhone
from utils.wrapperPhoneCSGM import wrapperPhoneCSGM
from utils.paramsCSGMPhone import paramsCSGMPhone

def main():
    pltFlag = 1
    dataFlag = 'phone'
    imname = '20180302_mmmm_6527_20180303T130940'
    pathData = r'/Users/leisongju/Documents/github/CCA-public/DP_data_example/google2019/test/'
    imgExt = '.png'
    flag_GT = True

    # 加载图像（此处假设 loadPhone 适用于该数据集）
    im_l, im_r, im_guide, gt_depth, conf_depth = loadPhone(pathData, imname)
    
    params = paramsCSGMPhone()
    dispar_map_pyr, _, _ = wrapperPhoneCSGM(im_l, im_r, im_guide, params)
    
    custom_colormap = plt.cm.jet
    
    if flag_GT:
        # 加载 GT 深度图（假设与文件名规则对应）
        gt_depth_img = cv2.imread(f"{pathData}/merged_depth/{imname}/result_merged_depth_center.png", cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(12,4))
        plt.subplot(1,4,1)
        plt.imshow(gt_depth_img, cmap=custom_colormap)
        plt.title('GT map')
        # 不同金字塔层次的视差图示例
        for i, dmap in enumerate(dispar_map_pyr):
            plt.subplot(1,4,i+2)
            if i == 0:
                resize_dispar = cv2.resize(dmap, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
            elif i == 1:
                resize_dispar = cv2.resize(dmap, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            else:
                resize_dispar = cv2.resize(dmap, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            plt.imshow(resize_dispar, cmap=custom_colormap, vmin=-1, vmax=1)
            plt.title(f'Pyr lvl {i+1}')
            plt.colorbar()
        plt.show()
    else:
        plt.figure()
        plt.imshow(-dispar_map_pyr[-1], cmap=custom_colormap, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Disparity map')
        plt.show()

if __name__ == '__main__':
    main()