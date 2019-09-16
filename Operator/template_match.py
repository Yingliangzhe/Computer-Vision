import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cannyEdge0_w.bmp')
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''
print(img.shape)
cv2.namedWindow('cannyEdge',0)

while(1):
	cv2.imshow('cannyEdge',img)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:  # hit escape to quit
		break
cv2.destroyAllWindows()
'''
# Hough circle detection
circles1 = cv2.HoughCircles(img1,cv2.HOUGH_GRADIENT,1,200,param1=20,param2=40,minRadius=200,maxRadius=300)
circles = circles1[0,:,:]#提取为二维
#circles = np.uint16(np.around(circles))#四舍五入，取整
for i in circles[:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,0,255),2)#画圆
    cv2.circle(img,(i[0],i[1]),1,(255,0,0),6)#画圆心

circle_y = circles[0][0].astype(int)
circle_x = circles[0][1].astype(int)

#img_1 = np.zeros((2076,3088),dtype=np.uint8)

#img_cut = img1[circle_y-1000-300:circle_y-1000, circle_x-398:circle_x+698+500]

#img_1[circle_y-1000-300:circle_y-1000, circle_x-398:circle_x+698+500] = img_cut


'''
# using hough probably transformation, but the result is not very well
minLineLength = 100
maxLineGap = 50
lines = cv2.HoughLinesP(img_1,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img_1,(x1,y1),(x2,y2),(0,255,0),2)
'''

#using lsd to detect line in segment
#lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_NONE)
#lines = lsd.detect(img)[0]
#draw_img = lsd.drawSegments(img,lines)

'''
lineLength = 1000
lines = cv2.HoughLines(img1, 1, np.pi/180, 300)
for line in lines:
    rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
    a = np.cos(theta)   #theta是弧度
    b = np.sin(theta)
    x0 = a * rho    #代表x = r * cos（theta）
    y0 = b * rho    #代表y = r * sin（theta）
    x1 = int(x0 + lineLength * (-b)) #计算直线起点横坐标
    y1 = int(y0 + lineLength * a)    #计算起始起点纵坐标
    x2 = int(x0 - lineLength * (-b)) #计算直线终点横坐标
    y2 = int(y0 - lineLength * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)    #点的坐标必须是元组，不能是列表。
'''

while(1):
    cv2.namedWindow('circle',0)
    cv2.imshow('circle',img)
    #cv2.namedWindow('cut',0)
    #cv2.imshow('cut',img_1)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # hit escape to quit
        break

cv2.destroyAllWindows()
#plt.imshow(img)
#plt.show()

