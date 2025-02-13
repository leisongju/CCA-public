"""
加载 DSLR 数据集图像，并进行预处理。
"""

import cv2
import numpy as np

def loadDSLRDP(pathData, imname, imgExt):
    # 构造左、右图路径，格式为 imname+'_L'+imgExt
    path_left = f"{pathData}{imname}_L{imgExt}"
    path_right = f"{pathData}{imname}_R{imgExt}"
    
    imL = cv2.imread(path_left)
    imR = cv2.imread(path_right)
    if imL is None or imR is None:
        raise ValueError("图像加载失败，请检查文件路径！")
    
    # 将图像归一化到 [0,1]，并转换为 RGB（opencv 默认 BGR）
    imL = cv2.cvtColor(imL, cv2.COLOR_BGR2RGB) / 255.0
    imR = cv2.cvtColor(imR, cv2.COLOR_BGR2RGB) / 255.0
    
    # 构造引导图（这里直接用左图乘以 255）
    im_guide = (imL * 255).astype('uint8')
    # 将左、右图转为灰度并乘以 255
    imL_gray = (cv2.cvtColor(imL, cv2.COLOR_RGB2GRAY) * 255).astype('uint8')
    imR_gray = (cv2.cvtColor(imR, cv2.COLOR_RGB2GRAY) * 255).astype('uint8')
    
    return imL_gray, imR_gray, im_guide