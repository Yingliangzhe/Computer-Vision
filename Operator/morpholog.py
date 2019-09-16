import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('flange_template_rectangle/TL.bmp',0)
img = cv2.imread('flange_template_square/LM.bmp',0)
#img = cv2.imread('corner_template_affine/TM_0.bmp',0)
#img = cv2.imread('cannyEdge0_bbgf.bmp')
#img = cv2.equalizeHist(img,None)

plt.imshow(img,cmap='gray')
plt.show()

#kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
str = input()


if str == 'erode':
	# 腐蚀
	erosion = cv2.erode(img, kernel, 1)

elif str == 'dilate':
	erosion = cv2.dilate(img, kernel, 1)

elif str == 'open':
	erosion = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

elif str == 'close':
	erosion = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

else:
	erosion = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
	erosion = cv2.erode(erosion, kernel, 1)

plt.figure()
plt.subplot(1,2,1),plt.imshow(img,'gray')#默认彩色，另一种彩色bgr
plt.subplot(1,2,2),plt.imshow(erosion,'gray')
plt.show()


#erosion = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
plt.figure()
plt.imshow(erosion,'gray')
plt.show()


erosion = cv2.Canny(erosion,10,30)
plt.figure()
plt.imshow(erosion,'gray')
plt.show()

pass

#cv2.imwrite('./cannyEdge0_bbgf_close.bmp',erosion)
