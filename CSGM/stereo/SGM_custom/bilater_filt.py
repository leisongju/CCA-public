# Code taken from : https://github.com/poolio/bilateral_solver
# Paper : The Fast Bilateral Solver
# Link : https://arxiv.org/abs/1511.03296

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg

def rgb2yuv(im):
    im = im.astype(np.float32)/255.0
    conversion_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]])
    im_yuv = im.dot(conversion_matrix.T)
    return im_yuv

MAX_VAL = 256

def get_valid_idx(valid, candidates):
    locs = np.searchsorted(valid, candidates)
    locs = np.clip(locs, 0, len(valid) - 1)
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs

class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        self.im = im
        im_yuv = rgb2yuv(im)
        Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] / sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        self.hash_vec = (MAX_VAL ** np.arange(self.dim))
        self._create_grid(coords_flat)
    
    def _create_grid(self, coords_flat):
        hashes = self._hash_coords(coords_flat)
        unique_hashes, inv_indices = np.unique(hashes, return_inverse=True)
        self.nvertices = len(unique_hashes)
        self.S = csr_matrix((np.ones(self.npixels), (np.arange(self.npixels), inv_indices)),
                            shape=(self.npixels, self.nvertices))
        self.blurs = []
        for d in range(self.dim):
            blur = csr_matrix((np.eye(self.nvertices)))
            self.blurs.append(blur)
    
    def _hash_coords(self, coord):
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)
    
    def splat(self, x):
        return self.S.T.dot(x)
    
    def slice(self, y):
        return self.S.dot(y)
    
    def blur(self, x):
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out
    
    def filter(self, x):
        return self.slice(self.blur(self.splat(x))) / self.slice(self.blur(self.splat(np.ones_like(x))))

def bistochastize(grid, maxiter=10):
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in range(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    m = n * grid.blur(n)
    return diags(m)

if __name__ == '__main__':
    import cv2
    im = cv2.imread('sample_image.jpg')
    grid = BilateralGrid(im)
    output = grid.filter(np.random.rand(im.shape[0]*im.shape[1]))
    print("双边滤波结果输出")