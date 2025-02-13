"""
计算手机数据下当前层次的 C-SGM 算法实现，
尽量复现 MATLAB 版本（包括双边滤波预处理、成本计算、二次拟合、先验融合以及 8 方向传播）。
"""
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(os.path.join(os.getcwd(), 'CSGM/DP_Phone/utils'))
# print("os.getcwd(): ",os.getcwd())
import numpy as np
import cv2
from CSGM.DP_DSLR.utils import disparCost  # 假设手机与 DSLR 使用相同的辅助函数，可以复用

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

def ensure_gray(im):
    """
    如果图像是彩色（多通道），则转换为灰度图；否则直接返回。
    """
    if len(im.shape) == 3:
        # 根据实际情况选择 COLOR_BGR2GRAY 或 COLOR_RGB2GRAY
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im

@timing
def currPhoneLvlCSGM(im_L, im_R, im_guide, params, prior_lvl_parab, prior_lvl_dispar):
    """
    参数说明与 DSLR 版本类似，视差范围通常略有调整（例如 dispRange=(-1, 1)）。
    此外对左右图像进行双边滤波降噪处理。

    输入:
      im_L, im_R: 灰度图（np.float64）
      im_guide: 引导图
      params: 参数字典
      prior_lvl_parab, prior_lvl_dispar: 上一层先验信息（首次为 None）
      
    输出:
      dispar_map: 子像素视差图 (H, W)
      sum_parab: 融合后的二次拟合参数字典
      conf_score_no_suprress: 置信度信息
    """
    # 判断是否为第一层
    flag_lvl = 1
    if prior_lvl_parab is not None:
        flag_lvl = 2

    # 对左右图像做双边滤波降噪（模拟 MATLAB 中滤波处理）
    im_L_orig = im_L.copy()
    im_R_orig = im_R.copy()
    im_L = im_L - cv2.bilateralFilter(im_L.astype(np.float32), d=15, sigmaColor=20, sigmaSpace=3)
    im_R = im_R - cv2.bilateralFilter(im_R.astype(np.float32), d=15, sigmaColor=20, sigmaSpace=3)

    # 设置视差候选值
    if flag_lvl == 1:
        dmin, dmax = params.get('dispRange', (-1, 1))
        params['dispar_vals'] = np.arange(dmin - 1, dmax + 2)
    else:
        min_disp = np.round(np.min(prior_lvl_dispar))
        max_disp = np.ceil(np.max(prior_lvl_dispar))
        params['dispar_vals'] = np.arange(min_disp - 2, max_disp + 3)
    
    # 根据插值方式进行处理
    if params.get('interpolant', 'default') != 'ENCC':
        cost_neig, conf_score, dispar_int_val, conf_score_no_suprress = disparCost.disparCost(im_L, im_R, params)
        parab = disparCost.genParab(cost_neig, dispar_int_val, params)
    else:
        cost_neig, conf_score, dispar_int_val, conf_score_no_suprress = disparCost.disparCost(im_L, im_R, params)
        parab, invalid_mask = disparCost.estSubPixENCC(cost_neig, dispar_int_val, im_L, im_R, params)
        conf_score[invalid_mask == 1] = 0.01**2
    
    # 置信度加权
    parab['a'] = parab['a'] * conf_score
    parab['b'] = parab['b'] * conf_score
    parab['c'] = parab['c'] * conf_score
    
    # 融合先验（多层处理时）
    if flag_lvl > 1:
        priorW = params.get('priorW', 0.05)
        prior_lvl_parab['a'] = cv2.resize(prior_lvl_parab['a'], (parab['a'].shape[1], parab['a'].shape[0]))
        prior_lvl_parab['b'] = cv2.resize(prior_lvl_parab['b'], (parab['a'].shape[1], parab['a'].shape[0]))
        prior_lvl_parab['c'] = cv2.resize(prior_lvl_parab['c'], (parab['a'].shape[1], parab['a'].shape[0]))
        parab['a'] = parab['a'] + priorW * prior_lvl_parab['a']
        parab['b'] = parab['b'] + priorW * prior_lvl_parab['b']
        parab['c'] = parab['c'] + priorW * prior_lvl_parab['c']
    
    parab = disparCost.refineParab(parab, params)
    sum_parab = disparCost.propCSGM(parab, im_guide, params)
    dispar_map = - sum_parab['b'] / (2.0 * sum_parab['a'])
    
    return dispar_map, sum_parab, conf_score_no_suprress

