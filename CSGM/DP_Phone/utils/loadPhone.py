"""
加载手机数据集图像，并进行中心裁剪预处理。
"""

import os
import cv2
import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    raise ImportError("读取EXR文件需要安装 OpenEXR 和 Imath，请运行：pip install OpenEXR Imath")

def cropCenter(img, target_size):
    """
    对输入图像 img 进行中心裁剪，裁剪尺寸为 target_size (height, width)。
    保留所有通道。
    """
    h, w = img.shape[:2]
    th, tw = target_size
    start_x = (w - tw) // 2
    start_y = (h - th) // 2
    return img[start_y:start_y+th, start_x:start_x+tw]

def readEXR(file_path):
    """
    使用 OpenEXR 库读取 EXR 文件，并返回单通道图像数据（float32类型）。
    """
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header['channels'].keys())
    if 'R' in channels:
        channel = 'R'
    else:
        channel = channels[0]
    # 读取指定通道的数据
    data_str = exr_file.channel(channel, FLOAT)
    data = np.frombuffer(data_str, dtype=np.float32)
    data.shape = (height, width)
    return data

def loadPhone(pathData, imname):
    """
    加载手机数据集图像，依据 MATLAB 版本实现：
      - 左右图按照 crop_l = (1344, 1008) 裁剪（灰度读取）
      - 引导图按照 crop_s = (672, 504) 裁剪后再放大 2 倍（彩色读取后可保持原信息）
      - 深度图（GT）和置信度图也按照 crop_s 裁剪，其中置信度图若为多通道则取第 1 个通道
    最后将各图转换为 float32 类型，返回 im_l, im_r, im_guide, gt_depth, conf_depth
    """
    crop_l = (1344, 1008)  # 高分辨率裁剪尺寸，用于左右图
    crop_s = (672, 504)    # 低分辨率裁剪尺寸，用于引导图、深度图、置信度图

    # 构造各个图像的文件路径
    file_path_right = os.path.join(pathData, "right_pd", imname, "result_rightPd_center.png")
    file_path_left  = os.path.join(pathData, "left_pd", imname, "result_leftPd_center.png")
    file_path_guide = os.path.join(pathData, "scaled_images", imname, "result_scaled_image_center.jpg")
    file_path_depth = os.path.join(pathData, "merged_depth", imname, "result_merged_depth_center.png")
    file_path_conf  = os.path.join(pathData, "merged_conf", imname, "result_merged_conf_center.exr")

    # 读取图像
    # MATLAB 对左右图直接读取灰度图，所以这里也指定灰度模式
    im_r = cv2.imread(file_path_right, cv2.IMREAD_GRAYSCALE)
    im_l = cv2.imread(file_path_left, cv2.IMREAD_GRAYSCALE)
    # 引导图使用彩色读取（后续裁剪时保持所有通道信息）
    im_guide = cv2.imread(file_path_guide, cv2.IMREAD_COLOR)
    im_guide = cv2.cvtColor(im_guide, cv2.COLOR_BGR2RGB)
    # 深度图读取（单通道或多通道均可，根据文件存储情况，一般为灰度图）
    gt_depth = cv2.imread(file_path_depth, cv2.IMREAD_ANYDEPTH)
    # 使用 OpenEXR 库读取置信度图（EXR 文件）
    conf_depth = readEXR(file_path_conf)
    
    # 确保置信度图为单通道（MATLAB 版本取第一通道）
    if conf_depth is not None and len(conf_depth.shape) == 3:
        conf_depth = conf_depth[:, :, 0]

    # 检查图像是否加载成功
    if im_r is None or im_l is None or im_guide is None or gt_depth is None or conf_depth is None:
        raise ValueError("图像加载失败，请检查路径！")

    # 按照 MATLAB 版本依次进行中心裁剪
    im_r = cropCenter(im_r, crop_l)
    im_l = cropCenter(im_l, crop_l)

    # 引导图先裁剪 crop_s，然后使用放大 2 倍
    im_guide = np.float32(cropCenter(im_guide, crop_s))
    im_guide = cv2.resize(im_guide, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gt_depth = cropCenter(gt_depth, crop_s)
    conf_depth = cropCenter(conf_depth, crop_s)

    # 转换为 float32 类型（MATLAB 中用 double()）
    im_r = im_r.astype(np.float32)
    im_l = im_l.astype(np.float32)
    im_guide = im_guide.astype(np.float32)
    gt_depth = gt_depth.astype(np.float32)
    conf_depth = conf_depth.astype(np.float32)

    return im_l, im_r, im_guide, gt_depth, conf_depth