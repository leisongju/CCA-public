"""
实现当前尺度下 C-SGM 处理：
1. 利用左右图像块匹配（采用绝对差聚合，使用 box filter 作为代价聚合窗口）计算 cost volume；
2. 对 cost volume 按照最佳代价选择初始离散视差；
3. 利用中心、左邻域和右邻域成本进行二次拟合，获得子像素级修正；
4. 根据修正得到最终视差图，同时返回拟合参数（a 和 b）。

注意：本实现假设输入图像为单通道灰度图（double 类型或 np.float64），
      参数中必须给出 'dispRange'（例如(-5,5)），以及 'encc_window' 表示成本聚合窗口大小。
"""

import numpy as np
import cv2
from CSGM.DP_DSLR.utils import disparCost

def compute_cost_volume(im_L, im_R, disp_candidates, window_size=5):
    """
    计算代价卷积：对每个候选视差 d，计算左图与右图（右图按照 d 平移后）的聚合绝对差（SAD）。
    
    参数：
       im_L: 左图，形状 (H, W)
       im_R: 右图，形状 (H, W)
       disp_candidates: 视差候选值数组（通常为一维整型或浮点数数组）
       window_size: 聚合窗口尺寸（例如 31）
       
    返回：
       cost_volume: 形状 (H, W, num_disp) 的代价卷积，每个通道对应一个视差候选值
    """
    H, W = im_L.shape
    num_disp = len(disp_candidates)
    # 初始代价设为较大值（不匹配区域）
    cost_volume = np.full((H, W, num_disp), 1e6, dtype=np.float64)
    
    # 对每个视差候选值，进行右图平移并计算绝对差，再用 box filter 进行聚合
    for k, d in enumerate(disp_candidates):
        shifted_R = np.full_like(im_R, 255.0)  # 用高灰度填充（使得边界成本较高）
        # 注意：本实现中视差候选值通常取负值（如 (-5,-4,...,-1)），
        # 对于 d < 0，则右图需要向右平移（即取 im_R[:, -d:W] 填入 shifted_R[:, :W-d]）
        if d < 0:
            dd = -d
            # 将右图从第 dd 列开始复制到左边（其余区域用高代价填充）
            shifted_R[:, :W - dd] = im_R[:, dd:W]
        elif d > 0:
            # d 正时，向左平移 d 个像素
            shifted_R[:, d:W] = im_R[:, :W - d]
        else:
            shifted_R = im_R.copy()

        # 计算逐像素绝对差
        abs_diff = np.abs(im_L - shifted_R)
        # 使用 box filter 聚合代价（转换为 float32 供 cv2.boxFilter 使用）
        abs_diff_f = abs_diff.astype(np.float32)
        aggregated = cv2.boxFilter(abs_diff_f, ddepth=-1, ksize=(window_size, window_size))
        cost_volume[:, :, k] = aggregated
    return cost_volume

def currLvlCSGM(im_L, im_R, im_guide, params, prior_lvl_parab, prior_lvl_dispar):
    """
    根据当前尺度（当前层次）下的左右图像，计算子像素视差图。

    参数：
       im_L, im_R: 左右图像（灰度图，np.float64，取值范围 [0, 255]）
       im_guide: 引导图（本实现中不直接使用，可用于后续滤波）
       params: 参数字典，必须包含：
              - 'dispRange': (min_disp, max_disp)（例如 (-5, -1)）
              - 'encc_window': 成本聚合窗口尺寸（如 31）
              - 'interpolant': 插值方法，例如 'ENCC' 或其他（例如 'f1'）
              - 'priorW': 先验权重（多层处理时使用）
       prior_lvl_parab, prior_lvl_dispar: 先验信息（多层处理中使用，首次传入为 None）
       
    返回：
       dispar_map: 子像素视差图，形状 (H, W)
       sum_parab: 字典，包含二次拟合系数 'a', 'b', 'c'（各形状均为 (H, W) 数组）
       conf_score_no_suprress: 当前未使用，返回成本计算中得到的置信度（可为 None或具体数组）
    """
    # 判断当前是否为第一层处理
    flag_lvl = 1
    if prior_lvl_parab is not None:
        flag_lvl = 2

    # 根据当前层级设置视差候选值数组
    if flag_lvl == 1:
        dmin, dmax = params.get('dispRange', (-10, 10))
        # MATLAB 中: params.dispar_vals = [dmin-1, dmin, ..., dmax+1]
        params['dispar_vals'] = np.arange(dmin - 1, dmax + 2)
    else:
        # 多层处理时：根据上一层粗估计的最小值和最大值扩展
        min_disp = np.round(np.min(prior_lvl_dispar))
        max_disp = np.ceil(np.max(prior_lvl_dispar))
        params['dispar_vals'] = np.arange(min_disp - 2, max_disp + 3)  # +3因为 np.arange 末端不包含

    # 根据插值方式选择计算路径
    if params.get('interpolant', 'default') != 'ENCC':
        # 调用 disparCost 计算成本、邻域、置信度和初始粗离散视差
        cost_neig, conf_score, dispar_int_val, conf_score_no_suprress = disparCost.disparCost(im_L, im_R, params)
        # 根据成本和离散视差进行二次拟合，生成拟合参数 parab（包含 a, b, c）
        parab = disparCost.genParab(cost_neig, dispar_int_val, params)
    else:
        # ENCC 插值方式
        cost_neig, conf_score, dispar_int_val, conf_score_no_suprress = disparCost.disparCost(im_L, im_R, params)
        parab, invalid_mask = disparCost.estSubPixENCC(cost_neig, dispar_int_val, im_L, im_R, params)
        # 对于无效像素，以极低置信度处理
        conf_score[invalid_mask == 1] = 0.01**2

    # 将置信度乘入每个二次拟合系数
    parab['a'] = parab['a'] * conf_score
    parab['b'] = parab['b'] * conf_score
    parab['c'] = parab['c'] * conf_score

    # 如果是多层处理，则融合上一层先验信息
    if flag_lvl > 1:
        priorW = params.get('priorW', 0.05)
        parab['a'] = parab['a'] + priorW * prior_lvl_parab['a']
        parab['b'] = parab['b'] + priorW * prior_lvl_parab['b']
        parab['c'] = parab['c'] + priorW * prior_lvl_parab['c']

    # 对二次拟合参数进行细化处理，修正低置信度及边缘区域
    parab = disparCost.refineParab(parab, params)

    # 成本传播：8方向传播后各方向结果求和
    sum_parab = disparCost.propCSGM(parab, im_guide, params)

    # 根据二次拟合公式计算子像素视差：x = -b/(2*a)
    dispar_map = - sum_parab['b'] / (2.0 * sum_parab['a'])

    return dispar_map, sum_parab, conf_score_no_suprress


# 注意：下面函数disparCost、genParab、estSubPixENCC、refineParab、propCSGM
# 均应由其他模块实现，并与 MATLAB 版本保持一致。