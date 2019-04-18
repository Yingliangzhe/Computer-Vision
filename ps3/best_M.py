# In this function we will calculate the best projection matrix
import numpy as np
import random
from least_squares_M_solver import *
from SVD_M_solver import *

def cal_res(pts_2d, pts_3d, M):
    pts_3d_num = pts_3d.shape[0]
    ones = np.ones((pts_3d_num, 4))
    ones[:,:-1] = pts_3d
    pts_3d = ones
    pts_2d_proj = np.dot(pts_3d, M.transpose())
    pts_2d_proj_im = pts_2d_proj[:pts_2d_proj.shape[0]] / pts_2d_proj[:, 2:3]
    pts_2d_proj_im = pts_2d_proj_im[:, :-1]
    res = np.linalg.norm(pts_2d_proj_im - pts_2d)
    return res

def best_M(pts_2d, pts_3d, constraint_num, test_pts_num, iteration):
    M = np.zeros((3,4),dtype=np.float32)
    pts_2d_num = pts_2d.shape[0]
    res = 1e9
    for i in range(iteration):
        # it's pssible this random function has the same index
        #index = np.random.randint(pts_2d_num, size=constraint_num)
        index = random.sample(range(pts_2d_num), constraint_num)
        M_temp, _ = least_squares_M_solver(pts_2d[index, :], pts_3d[index, :])

        # now calculate the residual
        test_index = [i for i in range(pts_2d_num) if i not in index]
        test_index = random.sample(test_index, test_pts_num)
        res_temp = cal_res(pts_2d, pts_3d, M_temp)
        if res_temp < res:
            res = res_temp
            M = M_temp

    return M, res


