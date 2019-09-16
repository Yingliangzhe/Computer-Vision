import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_g_12.5_pantex/view_point_1.bmp')
img_2 = cv2.imread('D:/Diplomarbeit/Bildsatz/Mono/handle_g_12.5_pantex/view_point_2.bmp')
SIFT = cv2.xfeatures2d_SIFT.create()
#SURF = cv2.xfeatures2d_SURF.create()

# 提取特征并计算描述子
kps1, des1 = cv2.xfeatures2d_SIFT.detectAndCompute(SIFT, img, None)
kps2, des2 = cv2.xfeatures2d_SIFT.detectAndCompute(SIFT, img_2, None)
#kps2, des2 = cv2.xfeatures2d_SURF.detectAndCompute(SURF, img, None)

# 新建一个空图像用于绘制特征点
img_sift = np.zeros(img.shape, np.uint8)
img_sift_2 = np.zeros(img_2.shape, np.uint8)
#img_surf = np.zeros(img.shape, np.uint8)
'''
# 绘制特征点
cv2.drawKeypoints(img, kps1, img_sift)
cv2.drawKeypoints(img_2, kps2, img_sift_2)
#cv2.drawKeypoints(img, kps2, img_surf)

# 展示
cv2.namedWindow("img", 0)
cv2.imshow("img", img)

cv2.namedWindow("sift", 0)
cv2.imshow("sift", img_sift)

cv2.namedWindow("sift2", 0)
cv2.imshow("sift2", img_sift_2)
cv2.waitKey(0)

#cv2.namedWindow("surf", 0)
#cv2.imshow("surf", img_surf)
#cv2.waitKey(0)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
'''
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.

img3 = cv2.drawMatchesKnn(img,kps1,img_2,kps2,matches,None,flags=2)

plt.imshow(img3),plt.show()