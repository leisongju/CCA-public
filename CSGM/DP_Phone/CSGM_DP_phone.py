#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSGM_DP_phone.py
~~~~~~~~~~~~~~~~
这是基于 MATLAB 原版 CSGM_DP_phone.m 改写的 Python 脚本，
针对 Dual-Pixel 手机数据集（例如 "Modeling Defocus-Disparity in Dual-Pixel Sensors" 数据）进行视差图计算与展示。
该脚本包括：
 1. 设置调试标志与数据路径；
 2. 加载指定图像（左、右、引导图，及 GT 与置信度信息）；
 3. 调用多层金字塔 CSGM 算法计算视差图；
 4. 使用 matplotlib 显示原始引导图、不同层次下的视差图与 GT；
 5. 对计算得到的视差图进行下采样后（0.5 倍）调用指标函数计算评价指标。

注意：部分处理（如 cmp_all_metrics_conf）目前作为占位符实现，
      请根据实际需求完善（如对比 MATLAB 中计算 SRCC 与几何均值的具体实现）。
      
@author:
@date:
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 从手机数据相关模块导入必要函数
from utils.loadPhone import loadPhone
from utils.wrapperPhoneCSGM import wrapperPhoneCSGM
from utils.paramsCSGMPhone import paramsCSGMPhone
from conf_calc_for_filter import conf_calc_for_filter
from filter_test_phone import BilateralGrid, BilateralSolver
import os
from skimage.io import imsave

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

def cmp_all_metrics_conf(disp, gt_depth, conf_depth, a, b, c):
    """
    占位符函数，用于计算各指标（例如：aiwe1, aiwe2, srcc, geomean）。
    实际实现中应详细计算视差图与地面真值间的指标，这里仅返回示例数值。
    """
    aiwe1val = 0.0
    aiwe2val = 0.0
    srccval = 0.0
    geomean = 0.0
    print("计算指标：(aiwe1, aiwe2, srcc, geomean) = ", aiwe1val, aiwe2val, srccval, geomean)
    return aiwe1val, aiwe2val, srccval, geomean

@timing
def main():
    # ==========================
    # 变量设置与数据标识
    # ==========================
    pltFlag = 1
    dataFlag = 'phone'
    
    # 指定待处理图像名称（可根据需求选择其他标注好的图像）
    # 下面给出几个示例（MATLAB 中为注释掉的选项），此处选择"Mail-box"示例：
    # imname = '20180319_mmmm_6527_20180318T122059'  # Orange
    # imname = '20180302_mmmm_6527_20180303T131130'  # plant
    # imname = '20180302_mmmm_6527_20180303T131633'  # Plant
    imname = '20180302_mmmm_6527_20180303T130940'   # Mail-box（示例：边缘区域问题）
    
    # 设置数据所在路径（请根据实际情况修改）
    pathData = r'/Users/leisongju/Documents/github/CCA/CCA-public/DP_data_example/google2019/test/'
    
    # ==========================
    # 加载手机数据集图像：左图、右图、引导图、GT 深度图和置信度图
    # ==========================
    # try:
    im_l, im_r, im_guide, gt_depth, conf_depth = loadPhone(pathData, imname)
    im_l = np.rot90(im_l)
    im_r = np.rot90(im_r)
    im_guide = np.rot90(im_guide)
    # except Exception as e:
        # print("图像加载出错：", e)
        # return

    # ==========================
    # 获取 CSGM 参数设置（针对手机数据集）
    # ==========================
    params = paramsCSGMPhone()
    
    # ==========================
    # 采用多层金字塔 CSGM 算法计算视差图
    # ==========================
    dispar_map_pyr, sum_parab, conf_score_no_suprress = wrapperPhoneCSGM(im_l, im_r, im_guide, params)
    conf_score = conf_calc_for_filter(dispar_map_pyr[-1], im_guide)
    # MATLAB 代码中将极小值截断
    conf_score[conf_score < 1e-20] = 1e-20
    conf_score = conf_score * conf_score_no_suprress[1][3]
    
    # 获取最终层（最高分辨率）细化后的视差图
    curr_dispar = dispar_map_pyr[-1]


    # 构建 bilateral grid（这里使用导引图作为 reference）
    grid_params = {
        'sigma_spatial': 8,
        'sigma_luma': 16,
        'sigma_chroma': 8
    }
    grid = BilateralGrid(im_guide, **grid_params)

    # 构造 BilateralSolver 参数
    bs_params = {
        'lam': 15,
        'A_diag_min': 1e-5,
        'cg_tol': 1e-5,
        'cg_maxiter': 40
    }

    # 将目标视差图展平为列向量（转换为 double 类型）
    t = dispar_map_pyr[-1].reshape(-1, 1).astype(np.double)
    # 以计算得到的置信度作为权重，展平后转换为 double 类型
    c = conf_score.reshape(-1, 1).astype(np.double)

    # 进行 bilateral solver 滤波
    output_solver = BilateralSolver(grid, bs_params).solve(t, c)
    # 重构成原图尺寸
    im_shape = im_guide.shape[:2]
    output_solver = output_solver.reshape(im_shape)
    # 归一化处理便于显示
    output_solver_norm = output_solver - np.min(output_solver)
    output_solver_norm = output_solver_norm / np.max(output_solver_norm)

    # 使用 matplotlib 显示各部分结果
    plt.figure(figsize=(10, 8))

    # 1. 显示导引图（Tgt）
    plt.subplot(2, 2, 1)
    plt.imshow(np.rot90(im_guide / 255.0, 3), cmap='gray')
    plt.title('Tgt')
    plt.axis('off')

    # 2. 显示下采样后的初步视差图（Est - Full and down-sample），采用 viridis colormap
    curr_dispar_dn = cv2.resize(dispar_map_pyr[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    plt.subplot(2, 2, 2)
    plt.imshow(np.rot90(curr_dispar_dn, 3), cmap='viridis')
    plt.title('Est - Full and down-sample')
    plt.axis('off')

    # 3. 显示 bilateral solver 滤波后的视差图（Filtered Disparity），采用 viridis colormap
    # output_solver_norm 为滤波后的结果归一化后的图像
    plt.subplot(2, 2, 3)
    plt.imshow(np.rot90(output_solver_norm, 3), cmap='viridis')
    plt.title('Filtered Disparity')
    plt.axis('off')

    # 4. 显示地面真值（GT），采用 viridis colormap
    plt.subplot(2, 2, 4)
    if gt_depth is not None:
        plt.imshow(gt_depth, cmap='viridis')
        plt.title('GT')
    else:
        plt.text(0.5, 0.5, 'GT not available', horizontalalignment='center', verticalalignment='center')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    # plt.close()




    # # ==========================
    # # 显示结果（对应 MATLAB 的 figure(8) 以及各 subplot）
    # # ==========================
    # plt.figure(figsize=(10, 8))
    
    # # 1. 显示引导图（Tgt）
    # plt.subplot(2, 2, 1)
    # plt.imshow(np.rot90(im_guide/255.0, 3), cmap='gray')
    # plt.title('Tgt')
    # plt.axis('off')
    
    # # 2. 显示上一层视差图（大致对应 MATLAB dispar_map{end-1}，粗估计）
    # if len(dispar_map_pyr) >= 2:
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(np.rot90(dispar_map_pyr[-2], 3), cmap='jet')
    #     plt.title('Est - Full (Coarse)')
    #     plt.axis('off')
    
    # # 3. 显示当前层细化后的视差图（dispar_map{end}）
    # plt.subplot(2, 2, 3)
    # plt.imshow(np.rot90(curr_dispar, 3), cmap='jet')
    # plt.title('Est - Full (Refined)')
    # plt.axis('off')
    
    # # 4. 显示地面真实深度（GT）
    # plt.subplot(2, 2, 4)
    # if gt_depth is not None:
    #     plt.imshow(gt_depth, cmap='jet')
    # else:
    #     plt.text(0.5, 0.5, 'GT not available', horizontalalignment='center', verticalalignment='center')
    # plt.title('GT')
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.show()
    
    # # ==========================
    # # 计算指标：先将当前视差图下采样（0.5 倍）
    # # ==========================
    # curr_dispar_dn = cv2.resize(curr_dispar, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # aiwe1val, aiwe2val, srccval, geomean = cmp_all_metrics_conf(np.rot90(curr_dispar_dn, 3),
    #                                                              gt_depth, conf_depth, 0, 0, 0)
    # print("SRCC:", srccval)
    
    # # ==========================
    # # 利用双边滤波对视差图进行平滑处理，并重新计算指标
    # # ==========================
    # dispar_filt = cv2.bilateralFilter(curr_dispar.astype(np.float32), d=5, sigmaColor=75, sigmaSpace=75)
    # dispar_filt = cv2.resize(dispar_filt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # aiwe1val2, aiwe2val2, srccval2, geomean2 = cmp_all_metrics_conf(np.rot90(dispar_filt, 3),
    #                                                                 gt_depth, conf_depth, 0, 0, 0)
    # print("Geomean values: (粗糙层 vs 平滑后) =", geomean, geomean2)
    
if __name__ == '__main__':
    
    main()