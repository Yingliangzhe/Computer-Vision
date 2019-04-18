import cv2
import numpy as np
import sys
import random
from collections import OrderedDict
# load_file has the same function as loadtxt, there is no need to write a seperate function
#from load_file import *
from least_squares_M_solver import *
from SVD_M_solver import *
from best_M import *

# M_norm matrix used to correct the projection matrix
M_norm_a = np.array([[-0.4583, 0.2947, 0.0139, -0.0040],
                     [0.0509, 0.0546, 0.5410, 0.0524],
                     [-0.1090, -0.1784, 0.0443, -0.5968]], dtype=np.float32)

isNormalized = True

def ps3_1_a(methode):
    # a) estimate camera projection matrix
    #pts_2d = load_file('input/pts2d-norm-pic_a.txt')
    #pts_3d = load_file('input/pts3d-norm.txt')
    pts_2d = np.loadtxt('input/pts2d-norm-pic_a.txt', dtype=float)
    pts_3d = np.loadtxt('input/pts3d-norm.txt', dtype=float)
    if methode == 'least_squares':
        # test results using a least squares solver
        M, res = least_squares_M_solver(pts_2d, pts_3d)
        ones = np.ones((pts_3d.shape[0], 4))
        # add a column of ones at the end of pts_3d matrix
        ones[:, :-1] = pts_3d
        pts_3d = ones
        pts_2d_proj = np.dot(pts_3d, M.transpose())
        # for all row in matrix pts_2d_proj doing this operation
        pts_2d_proj_im = pts_2d_proj[:pts_2d_proj.shape[0]] / pts_2d_proj[:, 2:3]
        # the last column is not used for residual calculation, because they are all 1
        pts_2d_proj_im = pts_2d_proj_im[:, :-1]
        res = np.linalg.norm(pts_2d_proj_im - pts_2d)
        print('Results with least squares:')
        print('M=\n%s' % M)
        print('Point set %s projected to point set %s' % (pts_2d, pts_2d_proj_im))
        print('Residual : %4f\n' % res)

    else:
        # b) using SVD methode to find the vector
        M = SVD_M_solver(pts_2d, pts_3d)
        ones = np.ones((pts_3d.shape[0], 4))
        # add a column of ones at the end of pts_3d matrix
        ones[:, :-1] = pts_3d
        pts_3d = ones
        pts_2d_proj = np.dot(pts_3d, M.transpose())
        # for all row in matrix pts_2d_proj doing this operation
        pts_2d_proj_im = pts_2d_proj[:pts_2d_proj.shape[0]] / pts_2d_proj[:, 2:3]
        pts_2d_proj_im = pts_2d_proj_im[:, :-1]
        res = np.linalg.norm(pts_2d_proj_im - pts_2d)
        print('Results with Singular Value Decomposition:')
        print('M=\n%s' % M)
        print('Point set %s projected to point set %s' % (pts_2d, pts_2d_proj_im))
        print('Residual : %4f\n' % res)


def ps3_1_b(isNormalized):
    if isNormalized:
        pts_2d = np.loadtxt('input/pts2d-norm-pic_a.txt', dtype=float)
        pts_3d = np.loadtxt('input/pts3d-norm.txt', dtype=float)
    else:
        pts_2d = np.loadtxt('input/pts2d-pic_b.txt', dtype=float)
        pts_3d = np.loadtxt('input/pts3d.txt', dtype=float)

    M_8, res_8 = best_M(pts_2d, pts_3d, constraint_num=8, test_pts_num=4, iteration=10)

    M_12, res_12 = best_M(pts_2d, pts_3d, constraint_num=12, test_pts_num=4, iteration=10)

    M_16, res_16 = best_M(pts_2d, pts_3d, constraint_num=16, test_pts_num=4, iteration=10)

    # this class is tuple
    residuals = (res_8, res_12, res_16)
    Ms = (M_8, M_12, M_16)
    res, M = min((res, M) for (res, M) in zip(residuals, Ms))
    print('Residuals:\nfor 8 pts: %.5f\nfor 12 pts: %.5f\nfor 16 pts: %.5f\n' % (
        res_8, res_12, res_16))
    print('Best Projection Matrix\nM =\n%s\n' % M)
    return M, res

def ps3_1_c():
    # estimate camera center position in the 3D world coordinates
    M,_ = ps3_1_b()
    Q = M[:, :3]
    m4 = M[:, 3]
    C = np.dot(-np.linalg.inv(Q), m4)
    print('Center of Camera = %s\n'%C)


if __name__ == '__main__':
    svd = 'svd'
    lst = 'least_squares'
    M, res = ps3_1_a(svd)
    pass


