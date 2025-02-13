"""
此模块提供了一系列辅助函数，主要用于：
  1. 计算左右图像之间的成本体（SAD 聚合）
  2. 利用成本值进行子像素二次拟合，得到拟合参数 a, b, c
  3. 若采用 ENCC 插值，则进行特殊的子像素处理
  4. 对二次拟合参数进行后处理（如边缘平滑、低信任区域抑制等）
  5. 实现 8 方向成本传播（简单使用均值滤波替代复杂 DP）
"""
from numba import njit

import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d  # 使用 1D 卷积


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

def imgaussfilt(image, sigma, filter_size=None, padding='symmetric'):
    """
    对图像 image 进行 2D 高斯滤波，使用两个 1D 卷积加速实现，
    复刻 MATLAB imgaussfilt(temp, sigma, 'Padding','symmetric') 的行为。
    
    参数:
      image       : 输入图像（支持灰度二维或彩色三维），建议转换为 np.float64 进行计算
      sigma       : 高斯核标准差（建议为标量，此处只处理各向同性情况）
      filter_size : 滤波器尺寸；如果为 None，则采用默认值 2*ceil(2*sigma)+1
      padding     : 填充类型，可选 'symmetric','replicate','circular'；若为数值，则表示常数填充的值
                    注意 MATLAB imgaussfilt(temp,...,'Padding','symmetric') 使用镜像扩充

    返回:
      filtered    : 滤波之后的图像，尺寸与原图一致
    """

    # 保证 sigma 为浮点数
    sigma = float(sigma)
    
    # 计算默认滤波器尺寸
    if filter_size is None:
        filter_size = int(2 * np.ceil(2 * sigma) + 1)
    # 保证 filter_size 为奇数
    if filter_size % 2 == 0:
        filter_size += 1
    pad = filter_size // 2

    # 生成 1D 高斯核
    x = np.arange(-pad, pad + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / np.sum(g)
    
    # 根据 padding 参数对图像进行预先填充
    if isinstance(padding, str):
        pad_mode = padding.lower()
        if pad_mode == 'symmetric':
            # 镜像扩充（边界元素会重复）
            padded = np.pad(image, pad_width=pad, mode='symmetric')
        elif pad_mode == 'replicate':
            # 边缘值延拓
            padded = np.pad(image, pad_width=pad, mode='edge')
        elif pad_mode == 'circular':
            padded = np.pad(image, pad_width=pad, mode='wrap')
        else:
            raise ValueError('Unsupported padding mode: {}'.format(padding))
    else:
        # 若 padding 为数值，则使用常数填充
        padded = np.pad(image, pad_width=pad, mode='constant', constant_values=padding)
    
    # 推荐在计算前将数据转换为 float64（MATLAB 内部往往转换为 double 进行卷积）
    padded = padded.astype(np.float64)
    
    # 根据图像维度分别处理
    if image.ndim == 2:
        H, W = image.shape
        # 第一步：沿水平方向使用 1D 卷积
        temp = convolve1d(padded, weights=g, axis=1, mode='constant', cval=0.0)
        # 切除左右填充区域，恢复为原图宽度
        temp = temp[:, pad: pad + W]
        # 第二步：沿垂直方向进行 1D 卷积
        temp = convolve1d(temp, weights=g, axis=0, mode='constant', cval=0.0)
        # 切除上下填充区域，恢复为原图高度
        filtered = temp[pad: pad + H, :]
    elif image.ndim == 3:
        H, W, C = image.shape
        filtered = np.empty_like(image, dtype=np.float64)
        for c in range(C):
            channel = padded[:, :, c]
            # 水平方向 1D 卷积
            temp = convolve1d(channel, weights=g, axis=1, mode='constant', cval=0.0)
            temp = temp[:, pad: pad + W]
            # 垂直方向 1D 卷积
            temp = convolve1d(temp, weights=g, axis=0, mode='constant', cval=0.0)
            filtered[:, :, c] = temp[pad: pad + H, :]
    else:
        raise ValueError("Unsupported image dimensions: only 2D or 3D images are supported.")
        
    return filtered


def imtranslate(image, shift, fill_value=0):
    """
    模仿 MATLAB imtranslate 函数：对图像进行平移（采用线性插值），
    shift 为 (dx, dy) 的元组，注意 MATLAB 中使用 [-d,0] 平移。
    """
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    rows, cols = image.shape[:2]
    return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value)

def imfilter(image, kernel, border_mode='replicate'):
    """
    模仿 MATLAB imfilter 对图像进行卷积。这里“same”效果通过 cv2.filter2D 实现，
    并根据 border_mode 选择边界处理（默认用 'replicate' 即 cv2.BORDER_REPLICATE）。
    """
    if border_mode == 'replicate':
        bmode = cv2.BORDER_REPLICATE
    else:
        bmode = cv2.BORDER_DEFAULT
    return cv2.filter2D(image, -1, kernel, borderType=bmode)

@timing
def disparCost(im_L, im_R, params):
    """
    Python 复刻 MATLAB disparCost.m 计算代价（cost）：
      输入：
        im_L, im_R：左、右图（预期为浮点型二维数组）
        params：参数字典，至少包含如下字段：
           'gaussKerSigma' : 高斯滤波的 sigma 值
           'cost'          : 代价类型，如 'SAD','SSD','CC','NCC','ZNCC','ENCC','BT','SAD_win'
           'dispar_vals'   : 迭代的视差值列表（例如一个 list 或 numpy 数组）
      输出：
        cost_neig         : 每个像素的最佳视差及左右邻域代价（h×w×3 数组）
        conf_score        : 最终用于缩放抛物线的置信度（h×w 数组）
        dispar_int_val    : 最优视差对应的整数值（h×w 数组）
        conf_score_no_suprress : 包含不同置信度计算中间输出（以字典模拟 MATLAB cell 数组）
    """
    gaussKerSigma = params['gaussKerSigma']
    defval = -1e-4
    mx = 2.2
    mn = 1
    conf_max = 1
    conf_min = 0.01

    # 确保输入为 float32
    im_L = im_L.astype(np.float32)
    im_R = im_R.astype(np.float32)
    
    disp_vals = np.array(params['dispar_vals'])
    num_disp = disp_vals.size
    H, W = im_L.shape

    # 为 score 分配空间，第三维度为不同视差的代价
    score = np.zeros((H, W, num_disp), dtype=np.float32)
    
    idx = 0
    cost_method = params['cost']

    if cost_method == 'SAD':  # 加权 SAD
        for d in disp_vals:
            # 注意平移方向与 MATLAB 相同，传入 (-d, 0)
            im_R_shift = imtranslate(im_R, (-d, 0), fill_value=0)
            temp = -np.abs(im_L - im_R_shift)
            score[:, :, idx] = imgaussfilt(temp, sigma=gaussKerSigma)
            idx += 1
    elif cost_method == 'SAD_win':  # 使用滑动窗口求 SAD
        # 构造平均滤波核 h，大小为 (gaussKerSigma, gaussKerSigma)
        kernel_size = int(gaussKerSigma)
        if kernel_size < 1:
            kernel_size = 1
        h_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        for d in disp_vals:
            im_R_shift = imtranslate(im_R, (-d, 0), fill_value=0)
            temp = -np.abs(im_L - im_R_shift)
            score[:, :, idx] = imfilter(temp, h_kernel, border_mode='replicate')
            idx += 1
    elif cost_method == 'SSD':  # 加权 SSD
        for d in disp_vals:
            im_R_shift = imtranslate(im_R, (-d, 0), fill_value=0)
            temp = -np.abs(im_L - im_R_shift) ** 2
            score[:, :, idx] = imgaussfilt(temp, sigma=gaussKerSigma, kernel_size=5)
            idx += 1
    elif cost_method == 'CC':  # 加权相关系数
        for d in disp_vals:
            im_R_shift = imtranslate(im_R, (-d, 0), fill_value=0)
            temp = im_L * im_R_shift
            score[:, :, idx] = imgaussfilt(temp, sigma=gaussKerSigma, kernel_size=5)
            idx += 1
    elif cost_method == 'NCC':  # 归一化互相关
        for d in disp_vals:
            im_R_shift = imtranslate(im_R, (-d, 0), fill_value=0)
            ene_norm = np.sqrt(
                imgaussfilt(im_R_shift * im_R_shift, sigma=gaussKerSigma, kernel_size=5) *
                imgaussfilt(im_L * im_L, sigma=gaussKerSigma, kernel_size=5)
            )
            # 防止除0
            ene_norm = np.maximum(ene_norm, abs(defval))
            score[:, :, idx] = imgaussfilt(im_L * im_R_shift, sigma=gaussKerSigma, kernel_size=5) / ene_norm
            idx += 1
    elif cost_method == 'ZNCC':  # 归一化互相关（零均值）
        im_L_mean = imgaussfilt(im_L, sigma=gaussKerSigma, kernel_size=5)
        im_L_ene = imgaussfilt((im_L - im_L_mean) ** 2, sigma=gaussKerSigma, kernel_size=5)
        for d in disp_vals:
            im_R_shift = imtranslate(im_R, (-d, 0), fill_value=0)
            im_R_shift_mean = imgaussfilt(im_R, sigma=gaussKerSigma, kernel_size=5)
            im_R_shift_ene = imgaussfilt((im_R_shift - im_R_shift_mean) ** 2, sigma=gaussKerSigma, kernel_size=5)
            normal_val = np.maximum(np.sqrt(im_R_shift_ene * im_L_ene), abs(defval))
            temp = imgaussfilt((im_L - im_L_ene) * imtranslate(im_R - im_R_shift_mean, (-d, 0), fill_value=0),
                               sigma=gaussKerSigma, kernel_size=5)
            score[:, :, idx] = temp / normal_val
            idx += 1
    elif cost_method == 'ENCC':  # 同 SAD
        for d in disp_vals:
            im_R_shift = imtranslate(im_R, (-d, 0), fill_value=0)
            temp = -np.abs(im_L - im_R_shift)
            score[:, :, idx] = imgaussfilt(temp, sigma=gaussKerSigma, kernel_size=5)
            idx += 1
    elif cost_method == 'BT':  # 双边阈值法
        kernel_size_kernel = int(gaussKerSigma)
        if kernel_size_kernel < 1:
            kernel_size_kernel = 1
        h_kernel = np.ones((kernel_size_kernel, kernel_size_kernel), dtype=np.float32) / (kernel_size_kernel ** 2)
        
        im_L_minus = (im_L + imtranslate(im_L, (-1, 0), fill_value=0)) / 2
        im_L_plus  = (im_L + imtranslate(im_L, (1, 0), fill_value=0)) / 2
        im_L_min = np.minimum(np.minimum(im_L_minus, im_L), im_L_plus)
        im_L_max = np.maximum(np.maximum(im_L_minus, im_L), im_L_plus)
        
        im_R_minus = (im_R + imtranslate(im_R, (-1, 0), fill_value=0)) / 2
        im_R_plus  = (im_R + imtranslate(im_R, (1, 0), fill_value=0)) / 2
        im_R_min = np.minimum(np.minimum(im_R_minus, im_R), im_R_plus)
        im_R_max = np.maximum(np.maximum(im_R_minus, im_R), im_R_plus)
        
        for d in disp_vals:
            shift_R_min = imtranslate(im_R_min, (-d, 0), fill_value=0)
            shift_R_max = imtranslate(im_R_max, (-d, 0), fill_value=0)
            shift_R = imtranslate(im_R, (-d, 0), fill_value=0)

            A1 = im_L - shift_R_max
            A2 = shift_R_min - im_L
            A  = np.maximum(0, np.maximum(A1, A2))
            
            B1 = shift_R - im_L_max
            B2 = im_L_min - shift_R
            B  = np.maximum(0, np.maximum(B1, B2))
            
            temp = np.minimum(A, B)
            score[:, :, idx] = imfilter(temp, h_kernel, border_mode='replicate')
            idx += 1
        score = -score
    else:
        raise ValueError("不支持的 cost 类型!")

    # 对 score 第三维度去除首尾（MATLAB中用 score(:,:,2:end-1)）
    score_center = score[:, :, 1:-1]  # shape: (H, W, num_disp-2)
    # 在最后一维降序排序，取前两大(注意：这里排序得到索引均是相对 score_center 的索引)
    sorted_indices = np.argsort(-score_center, axis=2)
    max_idx1 = sorted_indices[:, :, 0]  # 第一大值所在的索引（相对 score_center）
    max_idx2 = sorted_indices[:, :, 1]
    # 对应的分数：
    max_score1 = np.take_along_axis(score_center, np.expand_dims(max_idx1, axis=2), axis=2).squeeze(axis=2)
    max_score2 = np.take_along_axis(score_center, np.expand_dims(max_idx2, axis=2), axis=2).squeeze(axis=2)
    # 调整索引，加1恢复到 score 原始维度（因为 score_center = score[:,:,1:-1]）
    max_idx_full = max_idx1 + 1

    # 计算比值得分
    denom = np.minimum(max_score1, defval)
    rat_score = max_score2 / denom
    a_coef = 1.0 / (mx - mn)
    b_coef = -a_coef * mn
    conf_score = np.clip(a_coef * rat_score + b_coef, conf_min, conf_max) ** 2

    # 用字典模拟 MATLAB cell 数组存储置信度
    conf_score_no_suprress = dict()
    conf_score_no_suprress[2] = conf_score.copy()
    # 如果两个最大值的索引差小于等于1，则将置信度置为1
    diff_idx = np.abs(max_idx1 - max_idx2)
    conf_score[diff_idx <= 1] = 1
    conf_score_no_suprress[1] = conf_score.copy()

    # 另一种方式计算置信度（参考 GOOGLE 方法）
    r_0 = 0.6
    r_1 = 0.8
    eps_d = 0.5
    w_r = 3
    d_i1 = -max_score1
    d_i2 = -max_score2
    _in1 = np.maximum(d_i1, eps_d)
    _in1 = (_in1 - d_i2 * r_0) ** 2
    _in1 = _in1 / ((d_i2 ** 2) * ((r_1 - r_0) ** 2))
    _in1 = np.maximum(0, _in1)
    _in2 = np.minimum(_in1, 1)
    conf_score_no_suprress[3] = np.exp(-w_r * _in2)

    # 构造 cost_neig：选取对应视差索引的左右邻域（即 max_idx_full 的前后各1项）
    ys, xs = np.indices((H, W))
    idx_m1 = max_idx_full - 1
    idx_p1 = max_idx_full + 1
    cost_neig = np.zeros((H, W, 3), dtype=np.float32)
    cost_neig[:, :, 0] = score[ys, xs, idx_m1]
    cost_neig[:, :, 1] = score[ys, xs, max_idx_full]
    cost_neig[:, :, 2] = score[ys, xs, idx_p1]

    # 得到最佳视差对应的整型值
    dispar_int_val = np.array(params['dispar_vals'])[max_idx_full]

    return cost_neig, conf_score, dispar_int_val, conf_score_no_suprress




def subPixInterp(score_neig, params):
    """
    复刻 MATLAB 的 subPixInterp 函数，
    根据左右以及中间的成本值计算子像素精度的抛物线拟合参数。
    
    参数:
      score_neig : 大小为 (H, W, 3) 的 numpy 数组，分别代表左、中、右三个 cost 值。
      params     : 参数字典，必须包含：
          - 'interpolant': 插值方法，例如 'parabola', 'f1', 'f2', 'f3', 'f4', 'f5'
          - 'offset'     : 1D 数组，用于插值偏移
          - 'bias'       : 1D 数组，用于插值校正

    返回:
      parab      : 字典包含 'a', 'b', 'c'，各为 (H, W) 数组，对应抛物线参数
    """
    # 提取左右和中间成本
    left_val = score_neig[:, :, 0]
    mid_val = score_neig[:, :, 1]
    right_val = score_neig[:, :, 2]
    
    # 平移左右值，使其相对于中间值，并取反
    left_val = -(left_val - mid_val)
    right_val = -(right_val - mid_val)
    mid_val = mid_val - mid_val  # 全部置零
    
    # 将负值置为零（可能在视差边界处出现）
    left_val[left_val < 0] = 0
    right_val[right_val < 0] = 0
    
    # 记录左侧成本是否小于等于右侧成本
    left_smaller = left_val <= right_val
    
    # 根据插值方式选择插值函数 f
    interpolant = params.get('interpolant', 'parabola')
    if interpolant == 'parabola':
        f = lambda x: x / (x + 1)
    elif interpolant == 'f1':
        f = lambda x: 0.25 * (x + x ** 2)
    elif interpolant == 'f2':
        f = lambda x: 0.5 * (np.sin(x * np.pi / 2 - np.pi / 2) + 1)
    elif interpolant == 'f3':
        f = lambda x: 0.25 * (x + x ** 4)
    elif interpolant == 'f4':
        f = lambda x: np.maximum(0.25 * (x + x ** 4), 1 - np.cos(x * np.pi / 3))
    elif interpolant == 'f5':
        f = lambda x: 0.5 - 0.5 * np.cos(x * np.pi / 2)
    else:
        raise ValueError("Unsupported interpolant: {}".format(interpolant))
    
    # 定义两个候选函数
    eps_val = 1e-10  # 防止除零
    d_final_1 = lambda l, r: -0.5 + f(l / np.maximum(r, eps_val))
    d_final_2 = lambda l, r: 0.5 - f(r / np.maximum(l, eps_val))
    
    # 分别计算两个候选的子像素偏移
    x1 = d_final_1(left_val, right_val)
    x2 = d_final_2(left_val, right_val)
    
    # 根据 left_smaller 的情况选择偏移 x 值
    x = np.copy(x2)
    x[left_smaller] = x1[left_smaller]
    
    # 对于左右值均为零的位置，直接设为 0，避免 NaN
    mask = (left_val == 0) & (right_val == 0)
    x[mask] = 0
    
    # 对 x 进行校正——使用预先校准的 offset 与 bias
    # 使用 np.interp 对展开的 x 进行 1D 插值，并重塑为原尺寸
    biases = np.interp(x.ravel(), params['offset'], params['bias'])
    biases = biases.reshape(x.shape)
    x = x - biases
    
    # 构建抛物线参数
    # 还原左右值符号
    left_val = -left_val
    right_val = -right_val
    
    a = (left_val + right_val) / 2.0
    # 对于 x 为 NaN 的情况以及 a 为 0 的情况，赋予一个极小负数，防止后续运算问题
    a[np.isnan(x)] = -1e-50
    a[a == 0] = -1e-50
    b = x * (-2 * a)
    c = mid_val  # 此处 mid_val 全为零
    
    return {'a': a, 'b': b, 'c': c}


@timing
def genParab(score_neig, dispar_int_val, params):
    """
    Python 实现的 genParab 函数：
      利用 subPixInterp 得到子像素插值的抛物线参数，并根据整数视差 dispar_int_val 进行调整，
      使得所有抛物线的最小值均以零为中心。
      
    参数:
      score_neig      : 大小为 (H, W, 3) 的 numpy 数组，包含左右和中间的成本值。
      dispar_int_val  : 大小为 (H, W) 的 numpy 数组，表示整数视差值。
      params          : 参数字典，至少应包含 subPixInterp 所需要的字段，如 'interpolant', 'offset', 'bias'。
      
    返回:
      parab           : 字典包含 'a', 'b', 'c'，为调整后的抛物线参数。
    """
    # 首先调用 subPixInterp 得到初步抛物线参数
    parab = subPixInterp(score_neig, params)
    
    # 根据整数视差进行平移调整：将抛物线“平移”使最小值校正到 0
    a_new = parab['a']
    b_new = parab['b'] - 2 * parab['a'] * dispar_int_val
    c_new = parab['a'] * (dispar_int_val ** 2) - parab['b'] * dispar_int_val + parab['c']
    
    parab['a'] = a_new
    parab['b'] = b_new
    parab['c'] = c_new
    
    return parab

def refineParab(parab, params):
    """
    对子像素二次拟合参数进行后处理：
    1. 对低置信度区域（曲率较小区域）进行平坦化处理。
    2. 针对水平方向边界进行惩罚，即对边缘抛物线进行衰减。

    参数:
      parab: 字典，包含 'a', 'b', 'c'，各为 (H, W) 的 numpy 数组，
             表示初步计算得到的抛物线参数。
      params: 参数字典，至少应包含下列字段：
          - 'confidenceThresh': 置信度阈值（例如一个负值），用于判断低置信度区域
          - 'penalty_border'  : 边界惩罚因子（通常小于 1），控制边界平滑程度
          - 'border_len'      : 需要惩罚的边界宽度（单位为列数）
    
    返回:
      parab: 后处理后的抛物线参数（字典形式）
    """

    # ----- 第一步：去除低置信度曲线 -----
    # 定义低置信度的 a 对应的极小负值
    conf_thresh = params.get('confidenceThresh', -1e-3)
    def_val_a = conf_thresh * 1e-4
    # MATLAB 中判断条件为 parab.a > params.confidenceThresh，
    # 这里保持一致，对满足条件的位置进行平坦化处理
    idx_filter = parab['a'] > conf_thresh

    parab['a'][idx_filter] = def_val_a
    parab['b'][idx_filter] = 0
    parab['c'][idx_filter] = 0

    # ----- 第二步：对边界进行惩罚（X 方向） -----
    H, W = parab['a'].shape
    border_len = params.get('border_len', 10)
    penalty_border = params.get('penalty_border', 0.5)

    # 构造边界惩罚系数矩阵，初始全为 1
    border_penalty = np.ones((H, W), dtype=parab['a'].dtype)
    # 生成从 penalty_border 到 1 的线性序列，共 border_len 个数
    penalty_val = np.linspace(penalty_border, 1, border_len)

    # 左侧边界：每行前 border_len 个元素赋予 penalty_val（重复各行）
    border_penalty[:, :border_len] = np.tile(penalty_val, (H, 1))
    # 右侧边界：使用翻转后的 penalty_val 填充
    border_penalty[:, -border_len:] = np.fliplr(np.tile(penalty_val, (H, 1)))

    # 应用惩罚：对抛物线参数除以边界惩罚系数
    parab['a'] = parab['a'] / border_penalty
    parab['b'] = parab['b'] / border_penalty
    parab['c'] = parab['c'] / border_penalty

    return parab
@njit(cache=True)
def compute_propagated_parabola(a, b, c, P1, Ep):
    """
    MATLAB ComputePropagatedParabola 的实现：
      aNew = a - P1;
      bNew = b + 2 * P1 * Ep;
      cNew = c - P1 * (Ep^2);
    """
    a_new = a - P1
    b_new = b + 2 * P1 * Ep
    c_new = c - P1 * (Ep ** 2)
    return a_new, b_new, c_new
@njit(cache=True)
def compute_expected_value(prevA, prevB, eps=1e-10):
    """
    MATLAB ComputeExpectedValue 的实现：
      Ep = -prevB / (2*(prevA + eps));
    """
    return -prevB / (2 * (prevA + eps))

@njit(cache=True)
def compute_P_adaptive(Pedges, a, prevA, P1param):
    """
    MATLAB ComputePAdaptive 的实现：
      weightPrev = -prevA;
      weightCur = -a;      % （虽然后续并未使用 weightCur）
      weightPrev = clip(weightPrev, 0, 1);
      Padaptive = P1param .* Pedges .* weightPrev;
    """
    weightPrev = -prevA
    weightPrev = np.clip(weightPrev, 0, 1)
    Padaptive = P1param * Pedges * weightPrev
    return Padaptive

def propLine(im_guide, parab, params, dir_flag):
    """
    实现 MATLAB 的 propLine 函数：
      对 parabolas 进行水平（'H'）或垂直（'V'）方向的传播。
      
    参数:
      im_guide: 指导图像（支持灰度或彩色，若彩色则要求第三维为通道）。
      parab: 字典，包含 'a', 'b', 'c'（尺寸均为 (H, W)）。
      params: 参数字典，必须包含：
              'P1param'  — 标量
              'sigmaEdges' — 控制边缘保留的参数
      dir_flag: 'H' 表示水平方向，'V' 表示垂直方向（此时内部对 parabolas 进行转置）。
    返回:
      parab_prop: 传播后的抛物线参数字典，包含 'a','b','c'
    """
    P1param = params['P1param']
    sigmaEdges = params['sigmaEdges']
    
    # 计算相邻像素间的差分 df
    if im_guide.ndim == 3:
        df = np.sum(im_guide[:, 1:, :] - im_guide[:, :-1, :], axis=2) / im_guide.shape[2]
    else:
        df = im_guide[:, 1:] - im_guide[:, :-1]
    # MATLAB中 Pedges = [ones(size(im_guide(:,1))) exp(-df.^2./sigmaEdges^2)];
    ones_col = np.ones((im_guide.shape[0], 1), dtype=df.dtype)
    Pedges = np.concatenate((ones_col, np.exp(- (df ** 2) / (sigmaEdges ** 2))), axis=1)
    
    # 根据方向选择数据（对于垂直传播需要转置处理）
    if dir_flag == 'H':
        SaLR = parab['a'].copy()
        SbLR = parab['b'].copy()
        ScLR = parab['c'].copy()
        orig_a = parab['a'].copy()
    elif dir_flag == 'V':
        SaLR = parab['a'].T.copy()
        SbLR = parab['b'].T.copy()
        ScLR = parab['c'].T.copy()
        orig_a = parab['a'].T.copy()
    else:
        raise ValueError("dir_flag 必须为 'H' 或 'V'")
        
    H_line, W_line = SaLR.shape
    # 左向右传播
    for ii in range(1, W_line):
        prevA = SaLR[:, ii - 1]
        prevB = SbLR[:, ii - 1]
        Padaptive = compute_P_adaptive(Pedges[:, ii], orig_a[:, ii], prevA, P1param)
        Ep = compute_expected_value(prevA, prevB)
        a_col, b_col, c_col = compute_propagated_parabola(SaLR[:, ii], SbLR[:, ii], ScLR[:, ii], Padaptive, Ep)
        SaLR[:, ii] = a_col
        SbLR[:, ii] = b_col
        ScLR[:, ii] = c_col

    # 右向左传播
    if dir_flag == 'H':
        SaRL = parab['a'].copy()
        SbRL = parab['b'].copy()
        ScRL = parab['c'].copy()
        orig_a_rl = parab['a'].copy()
    elif dir_flag == 'V':
        SaRL = parab['a'].T.copy()
        SbRL = parab['b'].T.copy()
        ScRL = parab['c'].T.copy()
        orig_a_rl = parab['a'].T.copy()
    # 对比 MATLAB 采用 im_guide(:,1:end-1,:) - im_guide(:,2:end,:);
    if im_guide.ndim == 3:
        df_rl = np.sum(im_guide[:, :-1, :] - im_guide[:, 1:, :], axis=2) / im_guide.shape[2]
    else:
        df_rl = im_guide[:, :-1] - im_guide[:, 1:]
    ones_col_rl = np.ones((im_guide.shape[0], 1), dtype=df_rl.dtype)
    Pedges_rl = np.concatenate((np.exp(- (df_rl ** 2) / (sigmaEdges ** 2)), ones_col_rl), axis=1)
    
    for ii in range(W_line - 2, -1, -1):
        prevA = SaRL[:, ii + 1]
        prevB = SbRL[:, ii + 1]
        Padaptive = compute_P_adaptive(Pedges_rl[:, ii], orig_a_rl[:, ii], prevA, P1param)
        Ep = compute_expected_value(prevA, prevB)
        a_col, b_col, c_col = compute_propagated_parabola(SaRL[:, ii], SbRL[:, ii], ScRL[:, ii], Padaptive, Ep)
        SaRL[:, ii] = a_col
        SbRL[:, ii] = b_col
        ScRL[:, ii] = c_col
    
    # 合并左右两次传播
    if dir_flag == 'H':
        parab_prop = {
            'a': SaLR + SaRL,
            'b': SbLR + SbRL,
            'c': ScLR + ScRL
        }
    else:  # 'V'
        parab_prop = {
            'a': (SaLR + SaRL).T,
            'b': (SbLR + SbRL).T,
            'c': (ScLR + ScRL).T
        }
    return parab_prop

def propDiag(im_guide, parab, params, dir_flag):
    """
    实现 MATLAB 的 propDiag 函数：
      利用对角线分组传播 (包括主对角线（MD）和次对角线（OD）)
      
    参数:
      im_guide: 指导图像（2D 或 3D）
      parab: 字典，包含 'a','b','c'
      params: 参数字典，至少包含 'P1param' 和 'sigmaEdges'
      dir_flag: 'MD' 表示主对角线，'OD' 表示次对角线（使用 fliplr 翻转 im_guide）
    返回:
      一个字典，包含传播后的 'a','b','c'
    """
    P1param = params['P1param']
    sigmaEdges = params['sigmaEdges']
    
    if dir_flag == 'OD':
        im_guide = np.fliplr(im_guide)
        
    # 计算对角线方向的差分
    if im_guide.ndim == 3:
        df = np.sum(im_guide[1:, 1:, :] - im_guide[:-1, :-1, :], axis=2) / im_guide.shape[2]
    else:
        df = im_guide[1:, 1:] - im_guide[:-1, :-1]
    PedgesTopLeft = np.exp(- (df ** 2) / (sigmaEdges ** 2))
    # 补全左右边界：在最左边添加一列，在最上边添加一行
    ones_col = np.ones((PedgesTopLeft.shape[0], 1), dtype=PedgesTopLeft.dtype)
    PedgesTopLeft = np.concatenate((ones_col, PedgesTopLeft), axis=1)
    ones_row = np.ones((1, PedgesTopLeft.shape[1]), dtype=PedgesTopLeft.dtype)
    PedgesTopLeft = np.concatenate((ones_row, PedgesTopLeft), axis=0)
    
    if dir_flag == 'MD':
        SaLR = parab['a'].copy()
        SbLR = parab['b'].copy()
        ScLR = parab['c'].copy()
        orig_a = parab['a'].copy()
    elif dir_flag == 'OD':
        SaLR = np.fliplr(parab['a']).copy()
        SbLR = np.fliplr(parab['b']).copy()
        ScLR = np.fliplr(parab['c']).copy()
        orig_a = np.fliplr(parab['a']).copy()
    
    H, W = SaLR.shape
    # 对于对角线上（从左上向右下）传播，MATLAB循环 ii=2:size(SaLR,2)
    for ii in range(1, W):
        # 构造“前驱”向量：将当前列第一行与前一列的前 H-1 行组合
        prevATopLeft = np.concatenate((SaLR[0:1, ii:ii+1].flatten(), SaLR[:H-1, ii-1].flatten()))
        prevBTopLeft = np.concatenate((SbLR[0:1, ii:ii+1].flatten(), SbLR[:H-1, ii-1].flatten()))
        Padaptive = compute_P_adaptive(PedgesTopLeft[:, ii], orig_a[:, ii], prevATopLeft, P1param)
        Ep = compute_expected_value(prevATopLeft, prevBTopLeft)
        a_col, b_col, c_col = compute_propagated_parabola(SaLR[:, ii], SbLR[:, ii], ScLR[:, ii], Padaptive, Ep)
        SaLR[:, ii] = a_col
        SbLR[:, ii] = b_col
        ScLR[:, ii] = c_col

    if dir_flag == 'MD':
        SaRL = parab['a'].copy()
        SbRL = parab['b'].copy()
        ScRL = parab['c'].copy()
        orig_a_rl = parab['a'].copy()
    elif dir_flag == 'OD':
        SaRL = np.fliplr(parab['a']).copy()
        SbRL = np.fliplr(parab['b']).copy()
        ScRL = np.fliplr(parab['c']).copy()
        orig_a_rl = np.fliplr(parab['a']).copy()
    
    # 计算次对角线方向的差分
    if im_guide.ndim == 3:
        df2 = np.sum(im_guide[1:, 1:, :] - im_guide[:-1, :-1, :], axis=2) / im_guide.shape[2]
    else:
        df2 = im_guide[1:, 1:] - im_guide[:-1, :-1]
    PedgesBotLeft = np.exp(- (df2 ** 2) / (sigmaEdges ** 2))
    # 补全右边界和下边界：在最右边添加一列，在最下边添加一行
    ones_col_rl = np.ones((PedgesBotLeft.shape[0], 1), dtype=PedgesBotLeft.dtype)
    PedgesBotLeft = np.concatenate((PedgesBotLeft, ones_col_rl), axis=1)
    ones_row_rl = np.ones((1, PedgesBotLeft.shape[1]), dtype=PedgesBotLeft.dtype)
    PedgesBotLeft = np.concatenate((PedgesBotLeft, ones_row_rl), axis=0)
    
    for ii in range(1, W):
        # 构造“前驱”向量：取当前列前 H-1 行（下移一行）及当前列的最后一行来自前一列
        prevABotLeft = np.concatenate((SaRL[1:, ii-1].flatten(), np.array([SaRL[-1, ii]])))
        prevBBotLeft = np.concatenate((SbRL[1:, ii-1].flatten(), np.array([SbRL[-1, ii]])))
        Padaptive = compute_P_adaptive(PedgesBotLeft[:, ii], orig_a_rl[:, ii], prevABotLeft, P1param)
        Ep = compute_expected_value(prevABotLeft, prevBBotLeft)
        a_col, b_col, c_col = compute_propagated_parabola(SaRL[:, ii], SbRL[:, ii], ScRL[:, ii], Padaptive, Ep)
        SaRL[:, ii] = a_col
        SbRL[:, ii] = b_col
        ScRL[:, ii] = c_col

    if dir_flag == 'MD':
        a_prop = SaLR + SaRL
        b_prop = SbLR + SbRL
        c_prop = ScLR + ScRL
    elif dir_flag == 'OD':
        a_prop = np.fliplr(SaLR + SaRL)
        b_prop = np.fliplr(SbLR + SbRL)
        c_prop = np.fliplr(ScLR + ScRL)
        
    return {'a': a_prop, 'b': b_prop, 'c': c_prop}

@timing
def propCSGM(parab, im_guide, params):
    """
    复现 MATLAB 中的 propCSGM 函数：
      根据导引图 im_guide 计算水平、垂直和对角线方向的传播，
      叠加各方向结果得到最终传播后的二次拟合参数。
      
    参数:
      parab: 字典，包含 'a', 'b', 'c'（各为图像大小的数组）
      im_guide: 导引图，可为灰度图 (H, W) 或 RGB 图 (H, W, 3)。
      params: 参数字典，其中必须包含：
            - 'P1param'
            - 'sigmaEdges'
            - 'numIter': 列表或数组，存放各金字塔层的迭代次数
            - 'idx_pyr': 当前金字塔层的索引
    返回:
      sum_parab: 字典，包含传播后的 'a', 'b', 'c'
    """
    if im_guide.ndim == 3 and im_guide.shape[2] == 3:
        rot_im_guide = np.transpose(im_guide, (1, 0, 2))
    else:
        rot_im_guide = im_guide.T

    # 保存初始的 a（作为置信度），这里使用全 1（与 MATLAB 中 aInit = ones(size(parab.a)) 一致）
    aInit = np.ones_like(parab['a'])
    idx_pyr = params['idx_pyr'] - 1 if params['idx_pyr'] > 0 else 0
    numIter = params['numIter'][idx_pyr]

    if numIter > 0:
        sum_parab = {
            'a': np.zeros_like(parab['a']),
            'b': np.zeros_like(parab['a']),
            'c': np.zeros_like(parab['a'])
        }
    else:
        sum_parab = {
            'a': parab['a'].copy(),
            'b': parab['b'].copy(),
            'c': parab['c'].copy()
        }

    for iter in range(numIter):
        # 横向传播
        parab_h = propLine(im_guide, parab, params, 'H')
        # 竖向传播
        parab_v = propLine(rot_im_guide, parab, params, 'V')
        # 主对角线传播
        parab_MD = propDiag(im_guide, parab, params, 'MD')
        # 次对角线传播
        parab_OD = propDiag(im_guide, parab, params, 'OD')

        sum_parab['a'] = parab_h['a'] + parab_v['a'] + parab_MD['a'] + parab_OD['a']
        sum_parab['b'] = parab_h['b'] + parab_v['b'] + parab_MD['b'] + parab_OD['b']
        sum_parab['c'] = parab_h['c'] + parab_v['c'] + parab_MD['c'] + parab_OD['c']

        # 若未到最后一次迭代，则更新 parab 参数作为下一次传播的输入
        if iter != numIter - 1:
            numPaths = 8
            parab['a'] = (sum_parab['a'] / numPaths) * aInit
            parab['b'] = (sum_parab['b'] / numPaths) * aInit
            parab['c'] = (sum_parab['c'] / numPaths) * aInit
            # 此处 MATLAB 中也可调用 refineParab2 进行进一步平滑，这里保持一致则注释掉

    return sum_parab