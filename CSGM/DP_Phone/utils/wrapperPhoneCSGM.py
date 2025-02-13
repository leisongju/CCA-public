"""
封装手机数据集的 CSGM 算法实现，
支持多层金字塔处理，完全复现 MATLAB 版本的处理流程。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.currPhoneLvlCSGM import currPhoneLvlCSGM

def timing(func):
    """
    装饰器：计算并输出函数的运行时间
    """
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 耗时 {end_time - start_time:.6f} 秒")
        return result
    return wrapper

def imgaussfilt_sym_cv_sep(image, kernel_size=5, sigma=1):
    """
    用对称填充和 cv2.sepFilter2D 实现高斯滤波，
    模拟 MATLAB imgaussfilt 的边界处理（包括上下和左右一致）。

    参数:
      image       : 输入图像（二维灰度或三维彩色）
      kernel_size : 高斯核尺寸，默认为5
      sigma       : 高斯核标准差，默认为1

    返回:
      与原尺寸相同的滤波后的图像
    """
    pad = kernel_size // 2
    if image.ndim == 2:
        # 指定每个方向的pad
        padded = np.pad(image, pad_width=((pad, pad), (pad, pad)), mode='symmetric')
    elif image.ndim == 3:
        padded = np.pad(image, pad_width=((pad, pad), (pad, pad), (0,0)), mode='symmetric')
    else:
        raise ValueError("输入图像的维度不支持！")
    
    # 使用 OpenCV 自带的 1D 高斯核（分离卷积）
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
    
    # 利用分离卷积对填充后的图像进行滤波，注意指定 borderType 为 BORDER_CONSTANT，
    # 避免内部再次进行边界填充（我们已提前填充）
    filtered = cv2.sepFilter2D(padded, -1, kernel_1d, kernel_1d, borderType=cv2.BORDER_CONSTANT)
    
    # 裁剪掉填充部分，使结果与原图大小相同
    if image.ndim == 2:
        filtered = filtered[pad:-pad, pad:-pad]
    else:
        filtered = filtered[pad:-pad, pad:-pad, :]
    
    return filtered


def pyrImg(img, num_levels):
    """
    构建图像金字塔，并同时计算每一层的模糊图像与差分图。

    输入:
       img: 输入图像（numpy 数组）
       num_levels: 金字塔层数
    输出:
       im_blur: 各层高斯模糊后的图像列表
       im_diff: 各层原图与模糊图的差值列表
       im_pyr: 图像金字塔（cell 数组）, 其中最后一个元素为全分辨率图像
    """
    im_blur = []
    im_diff = []
    im_pyr = []
    current = img.copy()
    for level in range(num_levels):
        # 对当前图像进行高斯模糊
        # blurred = cv2.GaussianBlur(current, (5, 5), 1, borderType=cv2.BORDER_REFLECT)
        blurred = imgaussfilt_sym_cv_sep(current, 5, 1)
        diff = current - blurred
        im_blur.append(blurred)
        im_diff.append(diff)
        im_pyr.append(current)
        if level < num_levels - 1:
            current = cv2.pyrDown(current)
    # 反转列表，使得最后一个元素为全分辨率图像（模拟 MATLAB cell 数组排序）
    im_blur.reverse()
    im_diff.reverse()
    im_pyr.reverse()
    return im_blur, im_diff, im_pyr



@timing
def wrapperPhoneCSGM(im_L, im_R, im_guide, params):
    """
    复刻 MATLAB 版 wrapperPhoneCSGM.m 的功能：
      1. 预处理：生成左右视图和引导图的金字塔（同时计算模糊图和差分图）
      2. 如果启用 LRC 则校正左图并重构金字塔
      3. 根据 params['levels'] 执行单层或多层细化（包括上采样和拟合参数更新），
         调用 currPhoneLvlCSGM 计算每层的视差图。
      
    参数 params 应为字典，期望包含以下关键字段：
       'levels'       : 需要的层数
       'LRC_pyr_lvl'  : LRC金字塔层数
       'doLRC'        : 是否执行 LRC 校正（True/False）
       'LRC_level'    : 执行 LRC 时所用的金字塔层（1-based index）
       'plt_pyr'      : 是否调试显示金字塔各层 (1 表示显示)
       'pyr_max'      : 多层细化最大层数
       
    返回:
      dispar_map_pyr      : 各金字塔层对应的视差图列表
      sum_parab           : 最后一层的拟合参数字典（包含键 'a', 'b', 'c'）
      conf_score_no_suprress: 每层（或整体）的置信度输出
    """
    debug_flag = 1

    # 计算金字塔生成的层数
    num_lvl_pyr = max(params['levels'], params['LRC_pyr_lvl'])
    
    # 生成左右图像金字塔（同时计算模糊图和差分图）
    im_blur_L, im_diff_L, im_pyr_L = pyrImg(im_L, num_lvl_pyr)
    im_blur_R, im_diff_R, im_pyr_R = pyrImg(im_R, num_lvl_pyr)
    # 对引导图仅构造金字塔（忽略模糊与差分版本）
    _, _, im_pyr_guide = pyrImg(im_guide, num_lvl_pyr)
    
    # 若启用 LRC：按照指定 LRC 层进行辐射校正，再生成相应金字塔
    if params.get('doLRC', False):
        lrc_idx = params['LRC_level'] - 1  # MATLAB 1-based 转换为 Python 0-based
        denom = im_blur_L[lrc_idx] + im_diff_L[lrc_idx] + 1e-6
        LRCGain = (im_blur_R[lrc_idx] + im_diff_R[lrc_idx]) / denom
        # 将增益图调整为与 im_L 同样尺寸（cv2.resize 接收的尺寸顺序为：(width, height)）
        im_L_shape = (im_L.shape[1], im_L.shape[0])
        LRCGain_resized = cv2.resize(LRCGain, im_L_shape, interpolation=cv2.INTER_LINEAR)
        im_L_gain = im_L * LRCGain_resized
        # 重新生成校正后的左图金字塔
        _, _, im_L_gain_pyr = pyrImg(im_L_gain, num_lvl_pyr)
        im_pyr_L = im_L_gain_pyr

    dispar_map_pyr = []
    conf_score_no_suprress = []  # 若多层，则依次存储；若单层，直接赋值
    sum_parab = None  # sum_parab 的格式为字典，键 'a', 'b', 'c'
    
    if params['levels'] == 1:
        # 单层处理，使用金字塔中全分辨率图像（即列表最后一个元素）
        disparity_map, sum_parab, conf_score = currPhoneLvlCSGM(im_pyr_L[-1], im_pyr_R[-1], im_guide, params, None, None)
        dispar_map_pyr.append(disparity_map)
        conf_score_no_suprress = conf_score
    elif params['levels'] > 1:
        n_levels = len(im_pyr_L)
        # 初始层：对应 MATLAB 中 im_pyr{end-params.levels+1} (1-based) → Python 索引: n_levels - params['levels']
        start_index = n_levels - params['levels']
        disparity_map, sum_parab, conf_score = currPhoneLvlCSGM(im_pyr_L[start_index],
                                                                 im_pyr_R[start_index],
                                                                 im_pyr_guide[start_index],
                                                                 params, None, None)
        dispar_map_pyr.append(disparity_map)
        conf_score_no_suprress.append(conf_score)
        
        if params.get('plt_pyr', 0) == 1:
            plt.figure(201)
            plt.subplot(121); plt.imshow(disparity_map, cmap='jet'); plt.title('Disparity map - pyramid lvl: 1'); plt.colorbar();
            plt.subplot(122); plt.imshow(im_pyr_guide[start_index], cmap='gray'); plt.title('Guide image - pyramid lvl: 1'); plt.colorbar();
            plt.pause(1e-6)
        
        # 多层细化：后续每级用上一次的结果进行上采样和求解更新
        for idx_pyr in range(2, params['pyr_max'] + 1):
            params['idx_pyr'] = idx_pyr
            curr_index = n_levels - params['levels'] + idx_pyr - 1   # MATLAB: end-params.levels+idx_pyr (1-based转换为0-based)
            prev_index = curr_index - 1
            
            curr_lvl_shape = im_pyr_L[curr_index].shape[:2]  # (height, width)
            prev_lvl_shape = im_pyr_L[prev_index].shape[:2]
            # 计算当前层与前一层的尺度因子（通常接近2，但可能因取整略有差异）
            rescale_fact = curr_lvl_shape[0] / prev_lvl_shape[0]
            
            # 对上一层视差图进行双线性上采样，并乘以尺度因子
            dispar_init = cv2.resize(dispar_map_pyr[idx_pyr - 2],
                                     (curr_lvl_shape[1], curr_lvl_shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
            dispar_init = dispar_init * rescale_fact
            
            # 更新拟合参数：根据公式 a*(d/rescaleFact)^2 + b*(d/rescaleFact) + c
            sum_parab['a'] = sum_parab['a'] * ((1 / rescale_fact) ** 2)
            sum_parab['b'] = sum_parab['b'] * (1 / rescale_fact)
            sum_parab['a'] = cv2.resize(sum_parab['a'],
                                        (curr_lvl_shape[1], curr_lvl_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            sum_parab['b'] = cv2.resize(sum_parab['b'],
                                        (curr_lvl_shape[1], curr_lvl_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            sum_parab['c'] = cv2.resize(sum_parab['c'],
                                        (curr_lvl_shape[1], curr_lvl_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            
            # 进行下一层视差估计
            disparity_map, sum_parab, conf_score = currPhoneLvlCSGM(im_pyr_L[curr_index],
                                                                     im_pyr_R[curr_index],
                                                                     im_pyr_guide[curr_index],
                                                                     params, sum_parab, dispar_init)
            dispar_map_pyr.append(disparity_map)
            conf_score_no_suprress.append(conf_score)
            
            if params.get('plt_pyr', 0) == 1:
                plt.figure(200 + idx_pyr)
                plt.subplot(121); plt.imshow(disparity_map, cmap='jet'); plt.title('Disparity map - pyramid lvl: ' + str(idx_pyr)); plt.colorbar();
                plt.subplot(122); plt.imshow(im_pyr_guide[curr_index], cmap='gray'); plt.title('Guide image - pyramid lvl: ' + str(idx_pyr)); plt.colorbar();
                plt.pause(1e-6)
    
    return dispar_map_pyr, sum_parab, conf_score_no_suprress