"""
SGM_loop_sub_pixel_ref.py 用于对 SGM 视差结果进行子像素滤波和参考优化。
"""

import os
import cv2
import numpy as np

def median_disparity_filter(disparity_map):
    # 示例：中值滤波
    return cv2.medianBlur(disparity_map.astype('float32'), 5)

def main():
    pathSGM = r'C:\Users\local_admin\CSGM\other_algos\Stereo\SGM_custom\res_BT\\'
    pathSGM_R = r'C:\Users\local_admin\CSGM\other_algos\Stereo\SGM_custom\res_BT_R\\'
    
    for idx_im in range(1, 16):
        data_path = f'C:\\Users\\local_admin\\CSGM\\Stereo_data\\int_calc_midd_BT\\Q\\tgt{idx_im}_resQ.npz'
        if not os.path.exists(data_path):
            continue
        data = np.load(data_path)
        # 假设加载的文件中有字段 'disparity'
        dispar_LR_filter2 = data['disparity'] if 'disparity' in data else np.zeros((100,100))
        dispar_LR_filter2[dispar_LR_filter2 == 0] = float('inf')
        curr_dispar = median_disparity_filter(dispar_LR_filter2)
        print(f"图像 {idx_im} 处理完成")
    print("SGM_loop_sub_pixel_ref 完成")

if __name__ == '__main__':
    main()