"""
How2RunSGMWrapper.py 用于演示如何调用 SGMWrapper 计算视差图。
"""

import cv2
from SGMWrapper import SGMWrapper

def main():
    # 加载左右图像示例（灰度图）
    im_left = cv2.imread('image_left.png', cv2.IMREAD_GRAYSCALE)
    im_right = cv2.imread('image_right.png', cv2.IMREAD_GRAYSCALE)
    if im_left is None or im_right is None:
        print("图像加载失败")
        return
    disparity_map, C = SGMWrapper(im_left, im_right, params={'block_size': 5, 'disparity_range': (-16,16)})
    cv2.imwrite('disparity_map.tiff', disparity_map)
    print("视差图已保存")

if __name__ == '__main__':
    main()