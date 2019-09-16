import cv2
import numpy as np


#img = cv2.imread('cannyEdge0_bbgf.bmp')
img = cv2.imread('corner_edge/TM_0.bmp')
img_mor = cv2.imread('cannyEdge0_bbgf_close.bmp')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

points_pos = np.where(img_gray != 0)
points_pos = np.vstack((points_pos[0],points_pos[1]))

edge_points = [ ]

for i in range(len(points_pos[0])):
	points_tuple = (points_pos[1][i],points_pos[0][i])
	edge_points.append(points_tuple)
	cv2.circle(img_mor,edge_points[i],0,(0,0,255),-1)

cv2.imwrite('./overlap.bmp',img_mor)
cv2.namedWindow('redEdge',0)
cv2.imshow('redEdge',img_mor)
cv2.waitKey(0)
