"""
Stereo 图像匹配示例 2，加载中间数据，计算视差图并进行评价。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 假设加载中间数据文件（npz 格式）
    data_path = r'C:\Users\local_admin\CSGM\CSGM_Monin\stereo\int_disparity_calc\Q\tgt10_resQ.npz'
    data = np.load(data_path)
    # 假设数据中包含 im_guide_R, GT, mask, ndisp 等字段
    im_guide_R = data['im_guide_R']
    GT = data['GT']
    mask = data['mask']
    ndisp = data['ndisp']
    
    # 加载左右图（示例中作为已保存数据）
    imgL = data['im_L']
    imgR = data['im_R']
    
    window_size = 5
    min_disp = 0
    num_disp = 16
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(disparity * mask, cmap='jet')
    plt.title('原始视差图')
    
    # 使用 Fast Global Smoother Filter 进行平滑（要求 cv2.ximgproc 模块）
    if hasattr(cv2, 'ximgproc'):
        smoother = cv2.ximgproc.createFastGlobalSmootherFilter(im_guide_R, lambda_=10, sigmaColor=5,
                                                               lambdaAttenuation=0.25, numIters=10)
        confLeft = (disparity > -1).astype(np.float32)
        confLeft_filter = smoother.filter(confLeft)
        confLeft_filter[confLeft_filter == 0] = 1e-6
        Dfilter = smoother.filter(disparity * confLeft) / confLeft_filter
        plt.subplot(1,2,2)
        plt.imshow(Dfilter * mask, cmap='jet')
        plt.title('平滑视差图')
    else:
        print("cv2.ximgproc 模块未安装，无法使用 Fast Global Smoother Filter")
    
    plt.show()

if __name__ == '__main__':
    main()