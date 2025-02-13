"""
Stereo 图像匹配示例脚本，采用 OpenCV 计算视差图。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 加载左右图像
    imgL = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        print("图像加载失败")
        return
    
    # 使用 StereoBM
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    disparity = disparity.astype(np.float32) / 16.0
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(disparity, cmap='jet')
    plt.title('StereoBM 视差图')
    
    # 使用 StereoSGBM
    window_size = 5
    min_disp = 0
    num_disp = 16
    stereo_sgbm = cv2.StereoSGBM_create(minDisparity=min_disp,
                                        numDisparities=num_disp,
                                        blockSize=window_size,
                                        P1=8*3*window_size**2,
                                        P2=32*3*window_size**2,
                                        uniquenessRatio=10,
                                        speckleWindowSize=100,
                                        speckleRange=32)
    disparity2 = stereo_sgbm.compute(imgL, imgR)
    disparity2 = disparity2.astype(np.float32) / 16.0
    plt.subplot(1,2,2)
    plt.imshow(disparity2, cmap='jet')
    plt.title('StereoSGBM 视差图')
    plt.show()

if __name__ == '__main__':
    main()