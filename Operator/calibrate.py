import cv2
from createObjectPoint import *
from createChessboardPoint import *
import glob
import numpy as np

u_axis = 9
v_axis = 6
'''
def print_image_info(chessboard, chessboard_gray):
    # load the chessboard image
    # chessboard = cv2.imread('Calibration.jpg', cv2.IMREAD_GRAYSCALE)

    print('Type of  this image is:')
    print(chessboard.dtype)
    # shape (720, 1280, 3) color image
    print('shape of this image is:')
    print(chessboard.shape)
    print('Size of this image is:')
    print(chessboard.size)


    print('Type of  gray image is:')
    print(chessboard_gray.dtype)
    # shape (720, 1280) gray image
    print('shape of gray image is:')
    print(chessboard_gray.shape)
    print('Size of gray image is:')
    print(chessboard_gray.size)
    pass
'''

if __name__ == '__main__':

    # Arrays to store object points and image points from all the images
    object_points = []
    image_points = []
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    file_name_set = 'D:/Diplomarbeit/Bildsatz/TestGF_1/chessboard/'+'*.bmp'
    chessboard_list = sorted(glob.glob(file_name_set))

    print(len(chessboard_list))

    for counter in range(1,len(chessboard_list)+1):
        print('load the original image:')
        # -------------------------------------
        # ------------for webcam---------------
        # -------------------------------------
        #chessboard = cv2.imread('D:/Computer Vision/CameraCalibration/chessboardCalibration/chessboard_'+str(counter)+'.jpg')

        # -------------------------------------
        # ------------for webcam---------------
        # -------------------------------------
        chessboard = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/chessboard/'+str(counter)+'.bmp')

        ret, corners = cv2.findChessboardCorners(chessboard, (u_axis, v_axis), None)
        # this instruction makes the shape of corners into (54,2)
        corners = np.squeeze(corners, axis=1)
        np.array(corners)
        print(corners)

        if ret == True and counter <= len(chessboard_list):
            chessboard_copy = chessboard.copy()
            chessboard_copy = cv2.cvtColor(chessboard_copy,cv2.COLOR_BGR2GRAY)
            objp = createChessboardPoint(corners)
            object_points.append(objp)
            corners_subpixel = cv2.cornerSubPix(chessboard_copy,corners,(11,11),(-1,-1),criteria)
            image_points.append(corners_subpixel)
            # draw and display the corners
            cv2.drawChessboardCorners(chessboard, (u_axis, v_axis), corners_subpixel, ret)
            cv2.namedWindow('show Corners'+str(counter),0)
            cv2.imshow('show Corners'+str(counter), chessboard)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Calibrate Camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,image_points,chessboard_copy.shape[::-1],None,None)

    # undistort function
    #img2 = cv2.imread('D:/Computer Vision/CameraCalibration/chessboardCalibration/chessboard_1.jpg')
    img2 = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/chessboard/1.bmp')
    h, w = img2.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    dst = cv2.undistort(img2,mtx,dist,None,new_camera_matrix)
    cv2.imwrite('calibration_result.png',dst)

    #


    #print("x:\n",x)
    #print("y:\n",y)
    ## following line will make the coordinate to a

    #print(corners)
    #print(corners[0],'->',points_world[0])
    #print(corners[20],'->',points_world[20])
#plt.imshow(chessboard_gray, cmap='gray')
#plt.show()
#pass



