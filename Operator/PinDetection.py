import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import glob

def segmentLoadport(img_gray,window_radius,middle_x,middle_y):
	y_distance_1 = 700
	y_distance_2 = 1080
	x_distance = 950
	window_radius = int(window_radius)
	middle_y = int(middle_y)
	middle_x = int(middle_x)

	# comment in Diplomarbeit diary 3.Monat 11.09.2019
	if middle_y-y_distance_1-window_radius < 0:
		TL_pin = img_gray[0:middle_y-y_distance_1+window_radius,middle_x-x_distance-window_radius:middle_x-x_distance+window_radius]
		TL_pin_origin_initial = [0,middle_x-x_distance-window_radius]
	else:
		TL_pin = img_gray[middle_y - y_distance_1 - window_radius:middle_y - y_distance_1 + window_radius,
		         middle_x - x_distance - window_radius:middle_x - x_distance + window_radius]
		TL_pin_origin_initial = [middle_y - y_distance_1 - window_radius, middle_x - x_distance - window_radius]

	plt.figure('TL_pin')
	plt.imshow(TL_pin,cmap='gray')

	if middle_y-y_distance_1-window_radius < 0:
		TR_pin = img_gray[0:middle_y-y_distance_1+window_radius,middle_x+x_distance-window_radius:middle_x+x_distance+window_radius]
		TR_pin_origin_initial = [0,middle_x+x_distance-window_radius]
	else:
		TR_pin = img_gray[middle_y - y_distance_1 - window_radius:middle_y - y_distance_1 + window_radius,
		         middle_x + x_distance - window_radius:middle_x + x_distance + window_radius]
		TR_pin_origin_initial = [middle_y - y_distance_1 - window_radius, middle_x + x_distance - window_radius]

	plt.figure('TR_pin')
	plt.imshow(TR_pin,cmap='gray')

	if middle_y+y_distance_2-window_radius < 0:
		BM_pin = img_gray[0:middle_y + y_distance_2 + window_radius,
		         middle_x - window_radius:middle_x + window_radius]
		BM_pin_origin_initial = [0, middle_x - window_radius]
	else:
		BM_pin = img_gray[middle_y + y_distance_2 - window_radius:middle_y + y_distance_2 + window_radius,
		         middle_x - window_radius:middle_x + window_radius]
		print(BM_pin.shape)
		BM_pin_origin_initial = [middle_y + y_distance_2 - window_radius, middle_x - window_radius]

	plt.figure('BM_pin')
	plt.imshow(BM_pin,cmap='gray')

	plt.show()

	Pins_patch = {'TL_pin':TL_pin,'TR_pin':TR_pin,'BM_pin':BM_pin}

	Pins_patch_origin_initial = {'TL_pin':TL_pin_origin_initial,'TR_pin':TR_pin_origin_initial,'BM_pin':BM_pin_origin_initial}

	return Pins_patch,Pins_patch_origin_initial



def centerTemplateMatching(img_edge,template):

	template_gamma = exposure.adjust_gamma(template,0.35)
	template_filtered = cv2.bilateralFilter(template_gamma, 7, 75, 75)
	template_edge = cv2.Canny(template_filtered, 25,38, 3)

	cv2.namedWindow('flange',0)
	cv2.imshow('flange',img_edge)
	cv2.imshow('template',template_edge)
	cv2.waitKey(1)

	w,h = template_edge.shape[0:2][::-1]
	res = cv2.matchTemplate(img_edge,template_edge,cv2.TM_CCOEFF_NORMED)

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	cv2.rectangle(img_edge, top_left, bottom_right, 255, 2)

	middle_y = top_left[1] + h / 2
	middle_x = top_left[0] + w / 2
	window_radius = max(w / 2, h / 2)

	img_origin_initial = [top_left[1],top_left[0]] # convert to [y,x] format, and store in variable img_origin_initial
	center_window_size_y = h
	center_window_size_x = w

	plt.figure()
	#plt.subplot(121), plt.imshow(res, cmap='gray')
	#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	#plt.subplot(122), \
	plt.imshow(img_edge, cmap='gray')
	plt.title('Detected Point')
	plt.show()

	return middle_x,middle_y,window_radius,img_origin_initial,center_window_size_x,center_window_size_y

def loadPinTemplate(file_name_set):
	pin_templates = {}
	pin_list = sorted(glob.glob(file_name_set))

	for pin_name in pin_list:
		temp_pin_name = pin_name.split('\\')[-1].split('.')[-2]
		pin_img = cv2.imread(pin_name)
		single_pin_template = dict(zip([temp_pin_name],[pin_img]))
		pin_templates.update(single_pin_template)

	return pin_templates


def pinTemplateMatching(Pins_patch,pin_templates):
	# using template matching to locate the pin position
	pin_origin_patch = {}
	pin_image = {}

	for pin_patch_name in Pins_patch:
		for pin_name in pin_templates:
			if pin_patch_name in pin_name:

				pin_patch_image = Pins_patch[pin_patch_name]
				pin_patch_image_gamma = exposure.adjust_gamma(pin_patch_image,0.35)
				pin_patch_image_filtered = cv2.bilateralFilter(pin_patch_image_gamma,7,75,75)
				pin_patch_image_edge = cv2.Canny(pin_patch_image_filtered,19,30)

				plt.figure(pin_patch_name)
				plt.imshow(pin_patch_image_edge,cmap='gray')

				pin_template_img = pin_templates[pin_name]
				pin_template_img = cv2.cvtColor(pin_template_img,cv2.COLOR_BGR2GRAY)
				pin_template_img_gamma = exposure.adjust_gamma(pin_template_img,0.35)
				pin_template_img_filtered = cv2.bilateralFilter(pin_template_img_gamma,7,75,75)
				pin_template_img_edge = cv2.Canny(pin_template_img_filtered,19,40)

				plt.figure(pin_name)
				plt.imshow(pin_template_img_edge,cmap='gray')

				w,h = pin_template_img_edge.shape[0:2][::-1]

				res = cv2.matchTemplate(pin_patch_image_edge,pin_template_img_edge,cv2.TM_CCORR_NORMED)

				min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

				top_left = max_loc

				bottom_right = (top_left[0]+h,top_left[1]+w)

				cv2.rectangle(pin_patch_image_edge,top_left,bottom_right,255,2)

				plt.figure(pin_name)
				plt.subplot(221), plt.imshow(res, cmap='gray')
				plt.title('Matching Result')
				plt.subplot(222), plt.imshow(pin_patch_image_edge, cmap='gray')
				plt.title('Detected Point')

				top_left = list(top_left)
				top_left.reverse()
				top_left = tuple(top_left)
				single_pin_pos = dict(zip([pin_name],[top_left]))
				pin_origin_patch.update(single_pin_pos)

				single_clamp_image = dict(
					zip([pin_name], [pin_patch_image[top_left[0]:top_left[0] + h, top_left[1]:top_left[1] + w]]))
				pin_image.update(single_clamp_image)

				plt.subplot(212)
				plt.imshow(single_clamp_image[pin_name], cmap='gray')
				plt.show()

				break

	return pin_origin_patch,pin_image


def exactPinToInitial(pin_image,pin_origin_patch,Pins_patch_origin_initial):
	exact_pin_initial = {}
	for pin_name in pin_image:
		# store the pin origin into dictionary
		single_pin_origin_initial = dict(zip([pin_name], [np.array(pin_origin_patch[pin_name]) + np.array(Pins_patch_origin_initial[pin_name])]))
		exact_pin_initial.update(single_pin_origin_initial)

	return exact_pin_initial


def mserDetection():
	pass



img = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/LOADPORT/affine/5.png')
img_gamma = exposure.adjust_gamma(img,0.35)
img_gray = cv2.cvtColor(img_gamma,cv2.COLOR_BGR2GRAY)

img_edge = cv2.Canny(cv2.bilateralFilter(img_gray,5,75,75),20,50)



center_template = cv2.imread('Loadport_template/center.png')
center_template_gray = cv2.cvtColor(center_template,cv2.COLOR_BGR2GRAY)

middle_x,middle_y,window_radius,center_origin_initial,center_window_size_x,center_window_size_y = centerTemplateMatching(img_edge,center_template_gray)

#center_img = img_gray[300:1000,500:1000]
center_y_min = center_origin_initial[0]
center_y_max = center_origin_initial[0] + center_window_size_y
center_x_min = center_origin_initial[1]
center_x_max = center_origin_initial[1] + center_window_size_x
center_img = img_gray[center_y_min:center_y_max,center_x_min:center_x_max]

plt.figure()
plt.imshow(center_img,cmap='gray')
plt.show()

Pins_patch,Pins_patch_origin_initial = segmentLoadport(img_gray,window_radius,middle_x,middle_y)

file_name_set = './Loadport_template/' + '*.png'
pin_templates = loadPinTemplate(file_name_set)

pin_origin_patch,pin_image = pinTemplateMatching(Pins_patch,pin_templates)

#detection_methode = input()

#if detection_methode == 'MSER':
	#mserDetection()






pass