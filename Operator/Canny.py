import cv2
import sfm
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


# gamma correction in opencv
def gammaCorrection(img_original,gamma):
	lookUpTable = np.empty((1, 256), np.uint8)
	for i in range(256):
		lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
	res = cv2.LUT(img_original, lookUpTable)

	return res

# this function is needed for the createTrackbar step downstream
def nothing(x):
    pass


# read the experimental image
#img = cv2.imread('cannyEdge0_bbgf_close.bmp')
#img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex/view_point_2.bmp')
#img = cv2.imread('equhist_1_noreflec.bmp')
#img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_o/original_for_affine.bmp')
#img = cv2.imread('D:/Diplomarbeit/FOUP_flange_0.PNG')
#img = cv2.imread('binary0_wtgf.bmp')
#img = cv2.imread('Loadport.jpg')
#img = cv2.imread('corner_template_affine/RM_0.bmp')
#img = cv2.imread('histgram_0.bmp')
#img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex_reflec/blend_11_blicht_20ms_noreflec.bmp')
#kernel = np.array([[0, -1, 0],[0, 5, 0],[0, -1, 0]], np.float32)
#img = cv2.imread('flange_gf_template_square/TM.png')
#img = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/FOUP/view_point_h/8.png')
#img = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/LOADPORT/affine/1.png')
img = cv2.imread('Loadport_template/TR_pin.png')
#img = cv2.imread('circleTemplate.png')
img = exposure.adjust_gamma(img,0.35)


if img.shape[2] != 0:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#img = cv2.equalizeHist(img,None)
#img = cv2.filter2D(img,-1,kernel)
# create trackbar for canny edge detection threshold changes
#img = exposure.adjust_gamma(img,0.1)
#img = exposure.adjust_log(img,0.01)
#_,img = cv2.threshold(img,20,255,cv2.THRESH_BINARY)


cv2.namedWindow('canny',0)


# add ON/OFF switch to "canny"
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'canny', 0, 1, nothing)

# add lower and upper threshold slidebars to "canny"
cv2.createTrackbar('lower', 'canny', 0, 255, nothing)
cv2.createTrackbar('upper', 'canny', 0, 255, nothing)
cv2.createTrackbar('sigma', 'canny', 0, 21 , nothing)

sigma_old = 0
# Infinite loop until we hit the escape key on keyboard
while(1):

    # get current positions of four trackbars
    lower = cv2.getTrackbarPos('lower', 'canny')
    upper = cv2.getTrackbarPos('upper', 'canny')
    sigma = cv2.getTrackbarPos('sigma', 'canny')
    s = cv2.getTrackbarPos(switch, 'canny')


    if sigma == 0 or sigma == -1:
	    sigma = 1
    else:
        sigma = 2*sigma - 1
    sigma_new = sigma

    if s == 0:
        edges = img
    else:
        if sigma_old != sigma_new:
            #img_gaussian = cv2.GaussianBlur(img, (sigma_new, sigma_new), 0)
            img_filtered = cv2.bilateralFilter(img, sigma_new, 75, 75)
            sigma_old = sigma_new

        edges = cv2.Canny(img_filtered, lower, upper,3)

    # display images
    #cv2.namedWindow('original',0)
    #cv2.imshow('original', img)
    cv2.namedWindow('canny',0)
    cv2.imshow('canny', edges)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # hit escape to quit
        break

cv2.destroyAllWindows()

print('lower is '+str(lower))
print('upper is '+str(upper))
print('sigma is '+str(sigma))

Canny_edge = edges.copy()
print(Canny_edge.shape)
#cv2.imwrite('./cannyEdge_BM_bbgf.bmp',Canny_edge)
#cv2.imwrite('./corner_edge/TM_0.bmp',Canny_edge)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41))
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
#kernel = np.ones((17,17),np.uint8)
#kernel = np.eye(5,dtype=int)

str = input()


if str == 'erode':
	# 腐蚀
	erosion = cv2.erode(Canny_edge, kernel, 1)

elif str == 'dilate':
	erosion = cv2.dilate(Canny_edge, kernel, 1)

elif str == 'open':
	erosion = cv2.morphologyEx(Canny_edge,cv2.MORPH_OPEN,kernel)

elif str == 'close':
	erosion = cv2.morphologyEx(Canny_edge, cv2.MORPH_CLOSE, kernel)

else:
	erosion = cv2.morphologyEx(Canny_edge, cv2.MORPH_GRADIENT, kernel)
	erosion = cv2.erode(erosion, kernel, 1)

plt.figure()
plt.subplot(1,2,1),plt.imshow(Canny_edge,'gray')#默认彩色，另一种彩色bgr
plt.subplot(1,2,2),plt.imshow(erosion,'gray')
plt.show()
#cv2.imwrite('./corner_edge/LM_close.bmp',erosion)

'''
erosion = cv2.morphologyEx(Canny_edge,cv2.MORPH_OPEN, kernel)

plt.figure()
plt.subplot(1,2,1),plt.imshow(Canny_edge,'gray')#默认彩色，另一种彩色bgr
plt.subplot(1,2,2),plt.imshow(erosion,'gray')
plt.show()

pass
'''