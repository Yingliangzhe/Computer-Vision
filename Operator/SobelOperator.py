import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_wtgf_12.5_pantex/affine_1.bmp')

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

lower = 3
upper = 40

#edgesx8u = cv2.Canny(sobelx8u, lower, upper)
#edges_8u = cv2.Canny(sobel_8u,lower,upper)

'''
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()
'''

while(1):
    cv2.namedWindow('CV_64F',0)
    cv2.imshow('CV_64F',sobel_8u)
    cv2.namedWindow('CV_8U',0)
    cv2.imshow('CV_8U',sobelx8u)
    #cv2.namedWindow('cut',0)
    #cv2.imshow('cut',img_1)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # hit escape to quit
        break

cv2.imwrite('./sobelEdge1_wtgf_8.bmp',sobelx8u)
cv2.imwrite('./sobelEdge1_wtgf_64.bmp',sobel_8u)
