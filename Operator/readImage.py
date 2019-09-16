import cv2
import numpy as np
import matplotlib.pyplot as plt

img_0 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex/reflec_0.bmp')
img_1 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex/reflec_1.bmp')
#img_1 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_wb_12.5_pantex/view_point_1.bmp')
#img_2 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_wb_12.5_pantex/view_point_2.bmp')
#img_0 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex_reflec/blend_11_blicht_20ms.bmp')
#img_1 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_bbgf_12.5_pantex_reflec/blend_11_blicht_20ms_noreflec.bmp')

img_0 = cv2.cvtColor(img_0,cv2.COLOR_BGR2GRAY)
img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)

hist_0 = cv2.calcHist([img_0],[0],None,[256],[0,256])
hist_1 = cv2.calcHist([img_1],[0],None,[256],[0,256])

img_0_norm = cv2.normalize(img_0,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
img_1_norm = cv2.normalize(img_1,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)


plt.figure()
plt.title('img_0')
plt.subplot(121)
plt.imshow(img_0,cmap='gray')
plt.subplot(122)
plt.imshow(img_0_norm,cmap='gray')

plt.figure()
plt.title('img_1')
plt.subplot(121)
plt.imshow(img_1,cmap='gray')
plt.subplot(122)
plt.imshow(img_1_norm,cmap='gray')

plt.show()

img_0_equ = cv2.equalizeHist(img_0)
img_1_equ = cv2.equalizeHist(img_1)

hist_0_equ = cv2.calcHist([img_0_equ],[0],None,[256],[0,256])
hist_1_equ = cv2.calcHist([img_1_equ],[0],None,[256],[0,256])

plt.figure()
plt.title('img_0')
plt.subplot(121)
plt.plot(hist_0)
plt.subplot(122)
plt.imshow(img_0,cmap='gray')

plt.figure()
plt.title('img_1')
plt.subplot(121)
plt.plot(hist_1)
plt.subplot(122)
plt.imshow(img_1,cmap='gray')

plt.figure()
plt.title('img_0_equ')
plt.subplot(121)
plt.plot(hist_0_equ)
plt.subplot(122)
plt.imshow(img_0_equ,cmap='gray')

plt.figure()
plt.title('img_1_equ')
plt.subplot(121)
plt.plot(hist_1_equ)
plt.subplot(122)
plt.imshow(img_1_equ,cmap='gray')


mean_0,stddv_0 = cv2.meanStdDev(img_0)
mean_1,stddv_1 = cv2.meanStdDev(img_1)

print('mean of original image_0: ',mean_0)
print('mean of original image_1: ',mean_1)

mean_0_equ,_ = cv2.meanStdDev(img_0_equ)
mean_1_equ,_ = cv2.meanStdDev(img_1_equ)
print('mean of image_0 after equhist: ',mean_0_equ)
print('mean of image_1 after equhist: ',mean_1_equ)

cv2.imwrite('./equhist_0_reflec.bmp',img_0_equ)
cv2.imwrite('./equhist_1_noreflec.bmp',img_1_equ)

plt.show()

#cv2.imwrite('./histgram_0.bmp',img_0_equ)

#img_black = np.zeros([2076,3088,3],np.uint8)
#print(img_black.shape)
#cv2.namedWindow('black',0)
#cv2.rectangle(img_black,())
'''
while(1):
	cv2.namedWindow('black',0)
	cv2.imshow('black',img_0)
	cv2.namedWindow('equ',0)
	cv2.imshow('equ',img_0_equ)
	cv2.waitKey(0)
'''




