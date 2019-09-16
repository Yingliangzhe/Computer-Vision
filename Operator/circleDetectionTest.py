import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage import exposure

def templateMatchingCircleDetection(template,img_edge,img_name):

	#template = cv2.imread('circleTemplate.png')

	template_gamma = exposure.adjust_gamma(template,0.35)
	template_filtered = cv2.bilateralFilter(template_gamma, 7, 75, 75)
	template_edge = cv2.Canny(template_filtered, 25,38, 3)

	#cv2.namedWindow('flange',0)
	#cv2.imshow('flange',img_edge)
	#cv2.imshow('template',template_edge)
	#cv2.waitKey(1)

	w,h = template_edge.shape[0:2][::-1]
	res = cv2.matchTemplate(img_edge,template_edge,cv2.TM_CCOEFF_NORMED)

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	cv2.rectangle(img_edge, top_left, bottom_right, 255, 2)

	plt.figure(img_name)
	#plt.subplot(121), plt.imshow(res, cmap='gray')
	#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	#plt.subplot(122), \
	plt.imshow(img_edge, cmap='gray')
	plt.title('Detected Point')
	plt.show()

	middle_y = top_left[1]+h/2
	middle_x = top_left[0]+w/2
	window_radius = max(w/2,h/2)

	return middle_x,middle_y,window_radius

def loadTemplate(file_name_set):
	templates = {}
	corner_list = sorted(glob.glob(file_name_set))

	for img_name in corner_list:
		temp_name = img_name.split('\\')[-1].split('.')[-2]
		img = cv2.imread(img_name)
		single_template = dict(zip([temp_name], [img]))
		templates.update(single_template)

	return templates

file_name_set = 'D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/FOUP/rotation/'+ '*.png'

img_set = loadTemplate(file_name_set)
template = cv2.imread('D:/Computer Vision/Operator/SIFT/circleTemplate.png')

for img_name in img_set:

	img = img_set[img_name]
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img_gray_gamma = exposure.adjust_gamma(img_gray,0.35)
	img_edge = cv2.Canny(cv2.bilateralFilter(img_gray_gamma,7,75,75),25,34)

	templateMatchingCircleDetection(template,img_edge,img_name)

plt.show()

cv2.waitKey(1)