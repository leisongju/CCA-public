"""
计算 DSLR 数据下的置信度，用于调整视差图中平滑的部分。
该实现完全参考 MATLAB 版本的思路，主要分为三步：
  Step 1: 根据二次拟合参数 a 计算 c1；
  Step 2: 利用局部邻域像素差异来计算 c2，用于剔除边缘或大差异区域；
  Step 3: 利用 Sobel 算子和高斯滤波计算梯度，再得到 c3。
最后将三者结合得到整体的置信度。
"""

import numpy as np
import cv2

def calc_conf_dslr(sum_parab, im_guide, dispar_total, params):
    # 参数设置（根据 MATLAB 代码）
    sig_a = 5      # 控制 c1 的扩展程度
    sig_c = 512    # MATLAB 中用于累加（本例中未直接使用，可根据需要扩展）
    sig_u = 1      # 用于邻域差异计算（控制 c2）
    w_v = 125      # 用于计算梯度部分（控制 c3）
    e_v = 25       # 阈值

    # -----------------------------
    # Step 1: 计算 c1 —— 基于二次拟合参数中 a 的值
    # MATLAB 中 c1 = exp( log(abs(sum_parab.a)) / sig_a^2 )
    a_val = np.abs(sum_parab.get('a'))
    c1 = np.exp(np.log(a_val) / (sig_a ** 2))

    # -----------------------------
    # Step 2: 计算 c2 —— 检查邻域是否存在较大差异，剔除边缘区域
    # 这里模拟 MATLAB 中对每个像素遍历 3x3 邻域（中心除外）的作用
    # 为了边缘处理，先对 a_val 进行镜像复制（边界扩充）
    padded_a = cv2.copyMakeBorder(a_val, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)
    h, w = a_val.shape
    diff_max = np.zeros_like(a_val)
    # 对每个相邻偏移量计算差异
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = padded_a[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            diff = np.abs(a_val - shifted)
            diff_max = np.maximum(diff_max, diff)
    # 采用指数衰减作为 c2，差异越大，权重越低
    c2 = np.exp(-diff_max / sig_u)

    # -----------------------------
    # Step 3: 计算 c3 —— 利用图像梯度信息
    # 若 im_guide 为彩色图像，则先转换为灰度（MATLAB 版本中先归一化、转换后再乘以255，这里直接转换即可）
    if im_guide.ndim == 3:
        im_guide_gray = cv2.cvtColor((im_guide * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    else:
        im_guide_gray = im_guide
    # 使用 Sobel 算子计算水平和垂直方向的梯度
    sobel_x = cv2.Sobel(im_guide_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(im_guide_gray, cv2.CV_64F, 0, 1, ksize=3)
    grads = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grads = np.abs(grads)
    # 对梯度图进行高斯滤波（sigma 由参数 'gaussKerSigma' 指定，可与 MATLAB 保持一致）
    sigma = params.get('gaussKerSigma', 1)
    grads_filt = cv2.GaussianBlur(grads, (0, 0), sigmaX=sigma)
    # 计算按梯度反比加权部分 in3，然后利用指数函数得到 c3
    in3 = np.maximum(0, w_v / (grads_filt + 1e-6) - e_v)
    c3 = np.exp(-in3)

    # -----------------------------
    # 结合 c1, c2, c3 得到最终置信度
    conf = c1 * c2 * c3
    return conf
