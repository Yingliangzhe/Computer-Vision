import numpy as np

def least_squares_M_solver(pts_2d, pts_3d):
    pts_2d_num = pts_2d.shape[0]
    # initialization of matrix A and b(b is an array of zeros of right side)
    # 之前一直纠结这个11，本来想着应该是12列的，但是因为m23是1，所以把它移动到等式右面，正好是在image平面上的点
    A = np.zeros((2*pts_2d_num, 11), dtype=np.float32)
    b = np.zeros(2*pts_2d_num, dtype=np.float32)
    u = pts_2d[:, 0]
    v = pts_2d[:, 1]
    X = pts_3d[:, 0]
    Y = pts_3d[:, 1]
    Z = pts_3d[:, 2]
    # set the values of A Matrix and b vector
    zeros = np.zeros(pts_2d_num)
    ones = np.ones(pts_2d_num)
    # construct A matrix like lecture
    #
    A[0::2, :] = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -u*X, -u*Y, -u*Z))
    A[1::2, :] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones, -v*X, -v*Y, -v*Z))
    b[0::2] = u
    b[1::2] = v
    M, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.append(M,1) # add one element 1 at the end of this array
    M = M.reshape((3,4))
    return M, res







