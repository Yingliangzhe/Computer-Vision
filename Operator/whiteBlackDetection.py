import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_wtgf_12.5_pantex/affine_1.bmp')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

retval,dst = cv2.threshold(img_gray,60,90,cv2.THRESH_BINARY)
#ret= cv2.adaptiveThreshold(img_gray,150,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
cv2.imwrite('./binary1_wtgf.bmp',dst)

#img = cv2.imread('binary0_w.bmp')
#img = img.astype('uint8')


#find the contour of handling flange
'''
img_1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#_,contours,_= cv2.findContours(img_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
_,contours,_= cv2.findContours(img_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
cv2.drawContours(img,contours,-1,(0,0,255),3)
'''

while(1):
    cv2.namedWindow('binary',0)
    cv2.imshow('binary',dst)
    #cv2.namedWindow('cut',0)
    #cv2.imshow('cut',img_1)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # hit escape to quit
        break