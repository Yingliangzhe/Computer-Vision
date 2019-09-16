import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import glob
import math

img_edge = cv2.imread('corner_edge/LM_close.bmp')
#img_edge = cv2.imread('images/cannyEdge0_bbgf_close.bmp')


file_name_set = './flange_gf_template_square/' + '*.png'
templates = {}
corner_list = sorted(glob.glob(file_name_set))

for img_name in corner_list:
	temp_name = img_name.split('\\')[-1].split('.')[-2]
	img = cv2.imread(img_name)
	single_template = dict(zip([temp_name], [img]))
	templates.update(single_template)

print(img_edge.shape)
#img_ori = cv2.imread('corner_template_affine/TM_0.bmp')
#img_ori = cv2.imread('flange_template_square/TM.bmp')
#img_ori = cv2.imread('flange_template_rectangle/BM.bmp')
#img_ori = cv2.imread('flange_gf_template_square/TM.png')
#img_ori = cv2.imread('corner_edge/LM_close.bmp')



if img_edge.shape[2]!=0:
	img_edge_gray = cv2.cvtColor(img_edge,cv2.COLOR_BGR2GRAY)
'''
if img_ori.shape[2]!=0:
	img_ori_gray = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
'''
#img_ori_gray = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
#img_ori_gray = cv2.equalizeHist(img_ori_gray)
#img_ori_gray = cv2.bilateralFilter(img_ori_gray,5,50,5)
#img_ori_gray = exposure.adjust_gamma(img_ori_gray,0.05)
#img_ori_gray = cv2.equalizeHist(img_ori_gray)
#img_ori_gray = cv2.bilateralFilter(img_ori_gray,5,50,5)

'''
plt.figure()
plt.imshow(img_ori_gray,cmap='gray')
plt.show()
'''

def drawCrossLine(rho,theta,image):
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho
	x1 = int(x0 + 1000 * (-b))
	y1 = int(y0 + 1000 * (a))
	x2 = int(x0 - 1000 * (-b))
	y2 = int(y0 - 1000 * (a))

	cv2.line(img_edge, (x1, y1), (x2, y2), (255, 0, 0), 2)

detector = input()

if detector == 'houghL':
	# --------------------------------------------------------------------
	# ---------------------- sstandard hough line --------------------------
	# --------------------------------------------------------------------
	#img_edge_gray = cv2.bilateralFilter(img_edge_gray, 9, 75, 5)
	img_edge_gray = cv2.GaussianBlur(img_edge_gray, (5, 5), 0)
	img_edge_gray = cv2.Canny(img_edge_gray,5,13)




	plt.figure()
	# print(transformation+':'+img_name[img_gray.index(img)])
	plt.imshow(img_edge_gray, cmap='gray')
	plt.title('edge detection after morphological operation')
	plt.show()


	lines = cv2.HoughLines(img_edge_gray, 3, np.pi / 180, 100)
	# empty array to store the angle
	angle_l = []
	angle_l_deg = []
	rho_l = []

	angle_r = []
	angle_r_deg = []
	rho_r = []


	for x in range(len(lines)):
		for rho, theta in lines[x]:
			# this theta is rad, need to be converted into deg
			#if (theta*(180/np.pi)<50 and theta*(180/np.pi)>40) or (theta*(180/np.pi)<150 and theta*(180/np.pi)>130):
			if (theta*(180/np.pi)<140 and theta*(180/np.pi)>130):
				angle_l.append(theta)
				angle_l_deg.append(theta*(180/np.pi))
				rho_l.append(rho)

			if (theta*(180/np.pi)<60 and theta*(180/np.pi)>40):
				angle_r.append(theta)
				angle_r_deg.append(theta*(180/np.pi))
				rho_r.append(rho)

			if x == len(lines) - 1:
				angle_l_mean = np.mean(angle_l)
				angle_r_mean = np.mean(angle_r)
				rho_l_mean = np.mean(rho_l)
				rho_r_mean = np.mean(rho_r)

		drawCrossLine(rho,theta,img_edge)
		#plt.figure()
		# print(transformation+':'+img_name[img_gray.index(img)])
		#plt.imshow(img_edge, cmap='gray')


	drawCrossLine(rho_l_mean,angle_l_mean,img_edge)
	drawCrossLine(rho_r_mean,angle_r_mean,img_edge)




	plt.figure()
	# print(transformation+':'+img_name[img_gray.index(img)])
	plt.imshow(img_edge, cmap='gray')
	plt.title('standard hough line detection')




elif detector == 'houghP':
	# --------------------------------------------------------------------
	# -------------- probabilistic hough line segment --------------------
	# --------------------------------------------------------------------
	pass






elif detector == 'lsd':
	# --------------------------------------------------------------------
	# ------------------------------- LSD --------------------------------
	# --------------------------------------------------------------------
	#img_edge = cv2.Canny(img_ori_gray,6,14)
	#img_ori_gray = cv2.GaussianBlur(img_ori,(7,7),0)

	y_size = templates['TM'].shape[0]
	x_size = y_size

	lsd_detector = cv2.createLineSegmentDetector(0)

	for template_name in templates:
		img = templates[template_name]
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		#img = exposure.adjust_gamma(img,0.2)
		#img = cv2.equalizeHist(img,None)

		lines = lsd_detector.detect(img)[0]
		print(len(lines[0]))

		#-----------------------------------
		#    filter the short segment
		#-----------------------------------

		temp_line = []
		for line_index in range(0,len(lines)):
			print(lines[line_index])
			print(lines[line_index].shape)
			x_0 = lines[line_index][0][0]
			y_0 = lines[line_index][0][1]
			x_1 = lines[line_index][0][2]
			y_1 = lines[line_index][0][3]

			theta = (y_1-y_0)/(x_1-x_0)
			theta = np.arctan(theta)
			theta = theta*180/np.pi

			if (theta>30 and theta < 60) or (theta>-60 and theta<-30):
				if math.sqrt((y_1-y_0)**2+(x_1-x_0)**2) > 60:
					temp_line.append(lines[line_index])
			else:
				pass

		temp_line = np.array(temp_line)

		drawn_img_none = lsd_detector.drawSegments(img, temp_line)

		plt.figure(template_name)
		plt.subplot(2, 3, 1), plt.imshow(drawn_img_none, cmap='gray')
		plt.title('line in original')


		img_empty_none = np.zeros((y_size, x_size), np.uint8)
		drawn_on_empty_none = lsd_detector.drawSegments(img_empty_none,temp_line)
		edge_image_none = drawn_on_empty_none.copy()
		edge_image_none = cv2.cvtColor(edge_image_none,cv2.COLOR_BGR2GRAY)


		plt.subplot(2, 3, 4), plt.imshow(edge_image_none,cmap='gray')
		plt.title('edge image')

		#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
		#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))

		#erosion = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)

		# -----------------------------------------------------------

		img_hist = cv2.equalizeHist(img,None)
		img_hist = cv2.GaussianBlur(img_hist,(9,9),0)
		img_hist = exposure.adjust_gamma(img_hist,0.2)
		lines_hist = lsd_detector.detect(img_hist)[0]

		drawn_img_hist = lsd_detector.drawSegments(img_hist, lines_hist)

		plt.subplot(2, 3, 2), plt.imshow(drawn_img_hist, cmap='gray')  # 默认彩色，另一种彩色bgr
		plt.title('line in equhist')

		img_empty_hist = np.zeros((y_size, x_size), np.uint8)
		drawn_on_empty_hist = lsd_detector.drawSegments(img_empty_hist, lines_hist)
		edge_image_hist = drawn_on_empty_hist.copy()
		edge_image_hist = cv2.cvtColor(edge_image_hist, cv2.COLOR_BGR2GRAY)

		plt.subplot(2, 3, 5), plt.imshow(edge_image_hist, cmap='gray')
		plt.title('edge hist')

		# --------------------------------------------------------
		img_gamma = exposure.adjust_gamma(img, 0.2)
		#img_gamma = cv2.equalizeHist(img_gamma,0)

		lines_gamma = lsd_detector.detect(img_gamma)[0]

		drawn_img_gamma = lsd_detector.drawSegments(img_gamma, lines_gamma)

		plt.subplot(2, 3, 3), plt.imshow(drawn_img_gamma, cmap='gray')  # 默认彩色，另一种彩色bgr
		plt.title('line in gamma')

		img_empty_gamma = np.zeros((y_size, x_size), np.uint8)
		drawn_on_empty_gamma = lsd_detector.drawSegments(img_empty_gamma, lines_gamma)
		edge_image_gamma = drawn_on_empty_gamma.copy()
		edge_image_gamma = cv2.cvtColor(edge_image_gamma, cv2.COLOR_BGR2GRAY)

		plt.subplot(2, 3, 6), plt.imshow(edge_image_gamma, cmap='gray')
		plt.title('edge gamma')




	plt.show()


while(1):
	pass






