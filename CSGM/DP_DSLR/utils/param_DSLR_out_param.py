"""
生成 DSLR 输出参数，与 MATLAB 中参数设置相似。
"""

def param_DSLR_out_param(val_gauss, val_range, val_conf, val_p1, val_sig, val_lvl, val_priorW, val_iter,
                           val_cost, val_interpolant, val_encc_window):
    params = {}
    params['dispRange'] = (-val_range, val_range)
    params['numIter'] = [3, 3, 3]  # 默认迭代次数
    params['cost'] = val_cost
    params['gaussKerSigma'] = val_gauss
    params['confidenceThresh'] = val_conf
    params['interpolant'] = val_interpolant
    params['encc_window'] = val_encc_window
    # 其他参数可根据需要添加
    params['downSampleSize'] = [2.0, 4.0]
    return params