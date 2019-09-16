import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_o_12.5_pantex/view_point_1.bmp')

print(img.shape)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#plt.subplot(2, 2, 1)

img_gray = np.float32(img_gray)
dst_1 = cv2.cornerHarris(img_gray,3,3,0.06)
dst_1 = cv2.dilate(dst_1,None)
img[dst_1>0.01*dst_1.max()]=[255,0,0]

plt.title('origin')
plt.imshow(img)
plt.show()


