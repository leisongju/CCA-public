"""
加载 DSLR Qualitative 数据集图像，并进行中心裁剪预处理。
"""

import cv2
import numpy as np

def loadDSLRDP_Qual(pathData, imname, imgExt):
    path_left = f"{pathData}{imname}_L{imgExt}"
    path_right = f"{pathData}{imname}_R{imgExt}"
    
    imL = cv2.imread(path_left)
    imR = cv2.imread(path_right)
    if imL is None or imR is None:
        raise ValueError("图像加载失败，请检查文件路径！")
    
    imL = imL.astype('float64') / 255.0
    imR = imR.astype('float64') / 255.0
    
    clip = 100
    imL_cropped = imL[clip:-clip+1, clip:-clip+1]
    imR_cropped = imR[clip:-clip+1, clip:-clip+1]
    im_guide = (imL_cropped * 255).astype('uint8')
    
    imL_gray = (cv2.cvtColor(imL_cropped, cv2.COLOR_BGR2GRAY) * 255).astype('uint8')
    imR_gray = (cv2.cvtColor(imR_cropped, cv2.COLOR_BGR2GRAY) * 255).astype('uint8')
    
    return imL_gray, imR_gray, im_guide