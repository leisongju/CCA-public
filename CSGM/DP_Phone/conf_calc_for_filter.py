import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def conf_calc_for_filter(dispar, im_guide):
    """
    复刻 MATLAB 中 confCalcForFilter 函数：
      - 计算导引图的水平梯度，用于降低边缘处置信度；
      - 通过高斯滤波计算局部方差，降低纹理较弱区域的置信度；
      - 利用左右邻域视差差异，降低边缘处的置信度。
      
    参数:
      dispar: 视差图（单通道数组）。
      im_guide: 导引图（RGB 或灰度图像），通常数据范围在[0, 255]。
      
    返回:
      conf_score: 置信度图，与输入图像大小一致。
    """
    # 转换为灰度图
    if im_guide.ndim == 3 and im_guide.shape[2] == 3:
        # 使用 OpenCV 的转换，注意这里需要确保输入为 uint8 类型
        im_guide_uint8 = im_guide.astype(np.uint8)
        im_guide_gray = cv2.cvtColor(im_guide_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        im_guide_gray = im_guide.astype(np.float64)
    
    # MATLAB 里先做 im_guide/255 -> rgb2gray -> *255，相当于直接转换为灰度图
    
    # 计算水平方向梯度（使用 Sobel，ksize=3)
    Gx = cv2.Sobel(im_guide_gray, cv2.CV_64F, 1, 0, ksize=3)
    c1 = np.exp(-np.abs(Gx) / 50.0)
    
    # 计算局部方差（利用高斯平滑）
    w_v = 100.0
    eps_v = 0.5
    gauss_mean = gaussian_filter(im_guide_gray, sigma=9)
    gauss_mean_sq = gaussian_filter(im_guide_gray**2, sigma=9)
    var_guide = gauss_mean_sq - gauss_mean**2
    # 防止除0
    var_guide[var_guide < 1e-6] = 1e-6
    term = w_v / var_guide - eps_v
    term = np.maximum(0, term)
    c2 = np.exp(-term)
    
    # 计算左右邻域视差差异
    sig_u = 0.1
    d_i1 = dispar
    # 利用切片实现左右平移（边界用相邻值填充）
    d_i1_shift_left = np.zeros_like(d_i1)
    d_i1_shift_right = np.zeros_like(d_i1)
    d_i1_shift_left[:, 1:] = d_i1[:, :-1]
    d_i1_shift_left[:, 0] = d_i1[:, 0]
    d_i1_shift_right[:, :-1] = d_i1[:, 1:]
    d_i1_shift_right[:, -1] = d_i1[:, -1]
    
    in1_left = ((d_i1 - d_i1_shift_left) / (sig_u**2))**2
    in1_right = ((d_i1 - d_i1_shift_right) / (sig_u**2))**2
    in1 = np.minimum(in1_left, in1_right)
    c3 = np.exp(-in1)
    
    conf_score = c1 * c2 * c3
    return conf_score
