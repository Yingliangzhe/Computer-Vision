import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

def convlove(filter,mat,padding,strides):

    '''

    :param filter: convlove core, must be 2D
    :param mat: image
    :param padding: 对齐
    :param strides: step size
    :return: 返回卷积后的图片
    @author : Jialiang Yin
    '''

    '''
    result = None
    filter_size = filter.shape
    # the shape of a image ist (rows, columns, channels)
    mat_size = mat.shape
    if len(filter_size) == 2 and len(mat_size) == 2:
        if len(mat_size) == 2:
            ## especially for gray image
            channel = []   # initialization of channel
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')

            for j in range(0,mat_size[0],strides[1]):
                channel.append([])
                for k in range(0,mat_size[1],strides[0]):
                    # 这个语句实现了filter和image的矩阵元素的对应位置相乘
                    val = (filter * pad_mat[j * strides[1]:j * strides[1] + filter_size[0],
                                    k * strides[0]:k * strides[0] + filter_size[1]]).sum()
                    # 这个val做完乘法，又做了加法，把所有的元素的和加起来。所以应该是一个数字。
                    #if abs(val) >= 255:
                    #    val = 0
                    channel[-1].append(val)

            result = np.array(channel)

    return result
'''
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0, mat_size[0], strides[1]):
                    temp.append([])
                    for k in range(0, mat_size[1], strides[0]):
                        val = (filter * pad_mat[j * strides[1]:j * strides[1] + filter_size[0],
                                        k * strides[0]:k * strides[0] + filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j * strides[1]:j * strides[1] + filter_size[0],
                                    k * strides[0]:k * strides[0] + filter_size[1]]).sum()
                    channel[-1].append(val)

            result = np.array(channel)

    return result

if __name__ == '__main__':
    sobel_kernel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_kernel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    padding_size = [1,1,1,1]
    strides = [1,1]

    # load image and convert it to gray image
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

    img_gx = convlove(sobel_kernel_x,img,padding_size,strides)
    img_gy = convlove(sobel_kernel_y,img,padding_size,strides)

    img_gradiant = (img_gx**2+img_gy**2)**(1.0/2)
    # 因为在这个img_gradiant矩阵里，数字都是float型的，所以要先给它convert成uint8型，才能进行接下来的工作。
    img_gradiant = img_gradiant.astype(np.uint8)

    #img_gradiant = [[1:512],[1:500]]
    #cv2.imshow('image',img)
    cv2.imshow('image-gray', img_gradiant)
    cv2.waitKey(0)
    #plt.imshow(img_gradiant.astype(np.uint8),camp='gray')
    #plt.axis('off')
    #plt.show()


    # after loading a image using cv2.imread, the img1 is a matrix







