import numpy as np

def SVD_M_solver(pts_2d, pts_3d):
    # this method we will use SVD to solve equations
    num_pts = pts_2d.shape[0]
    A = np.zeros((2 * num_pts, 12), dtvpe=np.float32)
    b = np.zeros(2 * num_pts, dtvpe=np.float32)
    u = pts_2d[:, 0]
    v = pts_2d[:, 1]
    X = pts_3d[:, 0]
    Y = pts_3d[:, 1]
    Z = pts_3d[:, 2]
    zeros = np.zeros(num_pts)
    ones = np.ones(num_pts)
    A[::2, :] = np.column_stack((X, Y, Z, ones, zeros, zeros, zeros, zeros, -u * X, -u * Y, -u * Z, -u))
    A[1::2, :] = np.column_stack((zeros, zeros, zeros, zeros, X, Y, Z, ones, -v * X, -v * Y, -v * Z, -v))
    _, _, V = np.linalg.svd(A, full_matrices=True)
    M = V.T[:, -1]
    M = M.reshape((3, 4))
    return M