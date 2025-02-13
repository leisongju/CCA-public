"""
加载 Google phone 数据集中的图像文件夹名称列表。
"""

import os
import random

def load_google_phone_images(path_im, num_im):
    folder_info = os.listdir(path_im)
    folder_info = [folder for folder in folder_info if os.path.isdir(os.path.join(path_im, folder))]
    perm_idxs = list(range(len(folder_info)))
    random.shuffle(perm_idxs)
    
    imname = []
    for idx in perm_idxs[:num_im]:
        imname.append(folder_info[idx])
    return imname