import cv2
import numpy as np
import sys
import random
from collections import OrderedDict
# load_file has the same function as loadtxt, there is no need to write a seperate function
#from load_file import *
from least_squares_M_solver import *
from SVD_M_solver import *

# M_norm matrix used to correct the projection matrix
M_norm_a = np.array([[-0.4583, 0.2947, 0.0139, -0.0040],
                     [0.0509, 0.0546, 0.5410, 0.0524],
                     [-0.1090, -0.1784, 0.0443, -0.5968]], dtype=np.float32)

def ps3_1_a():
    # a) estimate camera projection matrix
    #pts_2d = load_file('input/pts2d-norm-pic_a.txt')
    #pts_3d = load_file('input/pts3d-norm.txt')
    pts_2d = np.loadtxt('input/pts2d-norm-pic_a.txt', dtype=float)
    pts_3d = np.loadtxt('input/pts3d-norm.txt', dtype=float)
    # test results using a least squares solver
    M, res = least_squares_M_solver(pts_2d, pts_3d)
    ones = np.ones((pts_3d.shape[0],4))
    # add a column of ones at the end of pts_3d matrix
    ones[:,:-1] = pts_3d
    pts_3d = ones
    pts_2d_proj = np.dot(pts_3d, M.transpose())
    # for all row in matrix pts_2d_proj doing this operation
    pts_2d_proj_im = pts_2d_proj[:pts_2d_proj.shape[0]]/pts_2d_proj[:,2:3]
    pts_2d_proj_im = pts_2d_proj_im[:, :-1]
    res = np.linalg.norm(pts_2d_proj_im - pts_2d)
    print('Results with least squares:')
    print('M=\n%s' % M)
    print('Point set %s projected to point set %s'%(pts_2d,pts_2d_proj_im))
    print('Residual : %4f\n'%res)
    # b) using SVD methode to find the vector
    M = SVD_M_solver(pts_2d,pts_3d)
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

def ps3_1_b():
    pass




if __name__ == '__main__':
    ps3_1_a()