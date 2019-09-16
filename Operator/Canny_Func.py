import cv2
import time

def nothing(x):
	pass

def Canny_func(** img):
	patch_edge = {}
	canny_param = {}

	print('before laod image')

	for patch in img.keys():
		if img[patch].shape[2] != 0:
			img[patch] = cv2.cvtColor(img[patch], cv2.COLOR_BGR2GRAY)

		# create trackbar for canny edge detection threshold changes
		cv2.namedWindow('canny', 0)
		print('there is a window')

		# add ON/OFF switch to "canny"
		switch = '0 : OFF \n1 : ON'
		cv2.createTrackbar(switch, 'canny', 0, 1, nothing)
		print('on/off switch')

		# add finish button
		finish = '0 : no \n1 : finish'
		cv2.createTrackbar(finish, 'canny', 0, 1, nothing)
		print('finish button')

		# add lower and upper threshold slidebars to "canny"
		cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
		print('lower trackbar')
		cv2.createTrackbar('upper', 'canny', 0, 255, nothing)
		print('upper trackbar')
		cv2.createTrackbar('sigma', 'canny', 0, 21, nothing)
		print('sigma trackbar')

		sigma_old = 0
		# Infinite loop until we hit the escape key on keyboard
		while (1):
			#print('in the loop')

			# get current positions of four trackbars
			lower = cv2.getTrackbarPos('lower', 'canny')
			upper = cv2.getTrackbarPos('upper', 'canny')
			sigma = cv2.getTrackbarPos('sigma', 'canny')
			s = cv2.getTrackbarPos(switch, 'canny')
			f = cv2.getTrackbarPos(finish,'canny')

			if sigma == 0 or sigma == -1:
				sigma = 1
			else:
				sigma = 2 * sigma - 1
			sigma_new = sigma

			if s == 0:
				edges = img[patch]
			else:
				if sigma_old != sigma_new:
					img_gaussian = cv2.GaussianBlur(img[patch], (sigma_new, sigma_new), 0)
					sigma_old = sigma_new

				edges = cv2.Canny(img_gaussian, lower, upper)

			# display images
			# cv2.namedWindow('original',0)
			# cv2.imshow('original', img)
			cv2.namedWindow('canny', 0)
			cv2.imshow('canny', edges)
			k = cv2.waitKey(1) & 0xFF
			#if k == 27:  # hit escape to quit
			#	break
			if f == 1:
				break

		# for every patch we can define a dictionary
		segment_egde = dict(zip([patch],[edges]))
		# using update function the dictionary can be stored in patch_edge
		patch_edge.update(segment_egde)

		canny_param_seg = dict(zip([patch], [[lower, upper]]))
		canny_param.update(canny_param_seg)

	return patch_edge,canny_param
