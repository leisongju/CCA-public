"""
生成手机数据集使用的 CSGM 参数，复现 MATLAB 版本中对参数的设置。
"""

def paramsCSGMPhone():
    params = {}
    # 对于手机数据，设置较宽的视差范围（例如 -5 到 -1），可根据实际情况调整
    params['dispRange'] = (-1, 1)
    params['numIter'] = [4, 4]
    params['levels'] = 2
    params['cost'] = 'SAD'
    params['gaussKerSigma'] = 11
    params['confidenceThresh'] = -0.0001
    # 原有下采样比例保持不变
    params['downSampleSize'] = [4.0/2, 8.0/2]
    # 新增：成本聚合窗口尺寸（与 MATLAB encc_window 类似，建议与图像块匹配窗口大小一致）
    params['encc_window'] = 17

    # 边界细化参数
    params['border_len'] = params['gaussKerSigma'] * 2 + max(abs(params['dispRange'][0]), abs(params['dispRange'][1]))
    params['penalty_border'] = 1000

    # 视差平滑惩罚参数
    params['P1param'] = 7
    params['sigmaEdges'] = 6

    # 金字塔处理参数
    params['plt_pyr'] = 0           # 是否绘制金字塔各层结果
    # levels 已设置为 [2]
    params['priorW'] = 0.01         # 多层先验权重，可根据实际情况调整
    params['idx_pyr'] = 1           # 第一层金字塔索引
    params['pyr_max'] = 2           # 金字塔最高层

    # ENCC 相关参数
    params['encc_window2'] = 17     # 第二个 ENCC 参数，与 first window 保持一致

    # 子像素插值参数
    params['interpolant'] = 'f1'    # 可选择 'f1' 或 'ENCC'
    params['bias'] = [3.34275500790682e-05, 0.00973533187061548, 0.0220652986317873, 0.0296091139316559,
                      0.0247546602040529, 2.39057189901359e-05, -0.0246540009975433, -0.0294469539076090,
                      -0.0218638572841883, -0.00956899579614401, 2.35768729908159e-05, 0.00971676409244537,
                      0.0220222137868404, 0.0295578259974718, 0.0247263666242361, 3.81323552574031e-05,
                      -0.0245587956160307, -0.0292639490216970, -0.0216563567519188, -0.00947358552366495,
                      0.000167622143635526]
    params['offset'] = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # LRC 参数（左右一致性检查）
    flagLRC = 1  # 可根据实际情况设定
    if flagLRC == 1:
        params['doLRC'] = True
        params['LRC_level'] = 1
        params['LRC_pyr_lvl'] = 4
    else:
        params['doLRC'] = False
        params['LRC_pyr_lvl'] = params['pyr_max']

    # 后处理滤波参数
    params['applyPostFilter'] = False
    params['guidedIter'] = 0
    params['fgsLambda'] = 50
    params['fgsSigmaColor'] = 3.5
    params['guidedDegreeOfSmoothing'] = 0.0001 * 256 * 256
    params['guidedNeighborhoodSize'] = [3, 3]

    return params