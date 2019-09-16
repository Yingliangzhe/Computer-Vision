import cv2
import numpy as np
import pylab as pl

img_edge = cv2.imread('LM_edge.bmp')
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_BGR2GRAY)

im2, contours, hierarchy = cv2.findContours(img_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(img_edge, contours[0], -1, (0,255,0), 3)



while(1) :
	cv2.imshow('edge contour',img_edge)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:  # hit escape to quit
		break