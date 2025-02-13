"""
SGM_loop.py 用于批量处理多幅图像，并保存 SGM 计算结果。
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    savepath = r'C:\Users\local_admin\CSGM\other_algos\Stereo\SGM_custom\res_BT\\'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # 自定义颜色映射，示例中使用 matplotlib jet
    import matplotlib.cm as cm
    CCM = cm.jet
    
    for idx_im in range(1, 16):
        data_path = f'C:\\Users\\local_admin\\CSGM\\Stereo_data\\int_calc_midd_BT\\Q\\tgt{idx_im}_resQ.npz'
        if not os.path.exists(data_path):
            continue
        data = np.load(data_path)
        print(f"处理图像 {idx_im}")
        # 此处可调用 SGMWrapper 或其他算法对图像进行视差计算，然后保存结果
    print("SGM_loop 完成")

if __name__ == '__main__':
    main()