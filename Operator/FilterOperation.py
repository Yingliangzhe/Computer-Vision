import cv2
import  numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex/affine_2.bmp',0)

def nothing(x):
	pass

#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('filter',0)

# add ON/OFF switch to "canny"
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'filter', 0, 1, nothing)

# add lower and upper threshold slidebars to "canny"
cv2.createTrackbar('diameter', 'filter', 0, 11, nothing)
cv2.createTrackbar('sigmaColor', 'filter', 0, 255, nothing)
cv2.createTrackbar('sigmaSpace', 'filter', 0, 255 , nothing)

while(1):
	diameter = cv2.getTrackbarPos('diameter','filter')
	sigma_color = cv2.getTrackbarPos('sigmaColor','filter')
	sigma_space = cv2.getTrackbarPos('sigmaSpace','filter')
	s = cv2.getTrackbarPos(switch,'filter')

	if s == 1:
		dst_bilateral = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
	else:
		dst_bilateral = img


	cv2.namedWindow('filter',0)
	cv2.imshow('filter',dst_bilateral)
	k = cv2.waitKey(1)&0xFF
	if k == 27:
		break



#filter_operation = input()

# filter2d filter used to enhence the contrast of image
#if filter_operation == 'mask':
	# kernel should be floating point type
kernel = np.array([[0, -1, 0],
                       [0, 5, 0],
                       [0, -1, 0]], np.float32)
dst_kernel = cv2.filter2D(img,-1,kernel)

# bilateral filter
#elif filter_operation == 'bilateral':
dst_bilateral = cv2.bilateralFilter(img,9,50,3)

# gaussian filter
#elif filter_operation == 'gaussian':
dst_gaussian = cv2.GaussianBlur(img, (9, 9), 0)



plt.subplot(121), plt.imshow(dst_gaussian, cmap='gray')
plt.title('gaussian image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst_bilateral, cmap='gray')
plt.title('bilateral image'), plt.xticks([]), plt.yticks([])

plt.show()

