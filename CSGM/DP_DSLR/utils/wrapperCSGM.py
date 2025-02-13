"""
封装 CSGM 算法，根据参数选择不同层次的处理，并支持多层金字塔的计算。
"""

import cv2
import numpy as np
from CSGM.DP_DSLR.utils.currLvlCSGM import currLvlCSGM

def build_pyramid(image, levels):
    """
    构建图像金字塔，返回一个列表，其中列表第一个元素为最粗层（最低分辨率），
    最后一个元素为原始图像。
    使用 cv2.pyrDown 构建金字塔。
    """
    pyr = []
    current = image.copy()
    for i in range(levels):
        if i > 0:
            current = cv2.pyrDown(current)
        pyr.append(current)
    # 将列表逆序，使得 pyr[0] 为最粗的层次（最低分辨率）
    pyr = pyr[::-1]
    return pyr

def wrapperCSGM(im_L, im_R, im_guide, params):
    """
    根据参数选择不同层次的 CSGM 处理。如果 params['levels'] > 1，则利用多层金字塔进行处理。
    
    输入:
        im_L: 左图
        im_R: 右图
        im_guide: 引导图
        params: 参数字典，必须包含 'levels' 字段，表示金字塔的层数
        
    输出:
        dispar_map_pyr: 一个列表，包含每个金字塔层次计算得到的视差图，
                          列表第一个元素为粗层（低分辨率估计），最后元素为最终细化结果。
        sum_parab: 当前层次的拟合参数
        conf_score_no_suprress: 置信度得分（具体计算在 currLvlCSGM 内部）
    """
    levels = params.get('levels', 1)
    dispar_map_pyr = []
    conf_score_no_suprress = None

    if levels > 1:
        # 构建左右图和引导图的金字塔
        pyr_L = build_pyramid(im_L, levels)
        pyr_R = build_pyramid(im_R, levels)
        pyr_guide = build_pyramid(im_guide, levels)

        # 更新参数中层数信息
        params['levels'] = levels

        # 第一层：最粗层，金字塔列表的第一个元素（最低分辨率）
        disp_coarse, sum_parab, conf_score_no_suprress = currLvlCSGM(pyr_L[0], pyr_R[0], pyr_guide[0], params, None, None)
        dispar_map_pyr.append(disp_coarse)

        # 从第二层开始逐层上采样和细化
        for lvl in range(1, levels):
            # 将上一层的视差图上采样到当前层尺寸（可以使用 cv2.resize 或 cv2.pyrUp）
            up_disp = cv2.resize(dispar_map_pyr[-1], (pyr_L[lvl].shape[1], pyr_L[lvl].shape[0]), interpolation=cv2.INTER_LINEAR)
            # 使用上一层的拟合参数和上采样视差作为先验信息，计算当前层的细化结果
            disp_refined, sum_parab, conf_score_no_suprress = currLvlCSGM(pyr_L[lvl], pyr_R[lvl], pyr_guide[lvl], params, sum_parab, up_disp)
            dispar_map_pyr.append(disp_refined)
    else:
        # 仅使用单层处理
        disp, sum_parab, conf_score_no_suprress = currLvlCSGM(im_L, im_R, im_guide, params, None, None)
        dispar_map_pyr.append(disp)

    return dispar_map_pyr, sum_parab, conf_score_no_suprress