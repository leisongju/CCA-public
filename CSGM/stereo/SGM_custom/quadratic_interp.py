"""
quadratic_interp.py 用于对 SGM 计算的成本进行二次拟合插值，提高子像素精度。
"""

import numpy as np
from scipy.interpolate import interp1d

def quadratic_interp(Cost_SGM, disparity_map, params, GT, mask, num_disp):
    tmpdisparity_map = disparity_map + 1
    tmpdisparity_map[tmpdisparity_map == 74] = 73
    tmpdisparity_map[tmpdisparity_map == 1] = 2
    
    h, w = Cost_SGM.shape[:2]
    cost_neig = np.zeros((h, w, 3))
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    indices1 = (ys.flatten(), xs.flatten(), (tmpdisparity_map - 1).flatten())
    indices2 = (ys.flatten(), xs.flatten(), tmpdisparity_map.flatten())
    indices3 = (ys.flatten(), xs.flatten(), (tmpdisparity_map + 1).flatten())
    
    cost_neig[:,:,0] = Cost_SGM[indices1].reshape(h, w)
    cost_neig[:,:,1] = Cost_SGM[indices2].reshape(h, w)
    cost_neig[:,:,2] = Cost_SGM[indices3].reshape(h, w)
    
    # 对每个像素进行二次拟合（示例代码）
    x = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            f = interp1d([-1, 0, 1], cost_neig[i, j, :], kind='quadratic')
            x[i, j] = f(0)
    adjusted_disparity = disparity_map + x
    return adjusted_disparity

def main():
    import cv2
    Cost_SGM = cv2.imread('Cost_SGM.png', cv2.IMREAD_GRAYSCALE)
    disparity_map = cv2.imread('disparity_map.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)
    params = {'offset': np.array([0, 1, 2]), 'bias': np.array([0, 0.1, 0.2])}
    GT = None
    mask = np.ones_like(disparity_map)
    num_disp = 64
    adjusted = quadratic_interp(Cost_SGM, disparity_map, params, GT, mask, num_disp)
    cv2.imwrite('adjusted_disparity.png', adjusted)
    print("二次插值完成")

if __name__ == '__main__':
    main()