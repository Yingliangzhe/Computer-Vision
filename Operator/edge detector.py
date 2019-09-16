import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure


img = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/FOUP/translation_h/6.png',0)
#img = cv2.imread('D:/Diplomarbeit/Bildsatz/TestGF_1/TestGF_anJialiang/Bilder_GF/FOUP/affine/2.png',0)

img = exposure.adjust_gamma(img,0.03)
blur = cv2.bilateralFilter(img,7,10,10)
# Laplace of Gaussian
laplacian = cv2.Laplacian(blur,cv2.CV_64F)

laplacian1 = laplacian/laplacian.max()

plt.figure()
plt.imshow(laplacian, cmap='gray')
plt.title('laplacian'), plt.xticks([]), plt.yticks([])


#plt.show()

#canny = cv2.Canny(laplacian,8,30,3)
# sobel operator
gradient_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
gradient_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(gradient_x) # 转回uint8
absY = cv2.convertScaleAbs(gradient_y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

plt.figure()
plt.imshow(dst, cmap='gray')
plt.title('sobel'), plt.xticks([]), plt.yticks([])









plt.show()
pass