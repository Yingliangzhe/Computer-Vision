# Image processing notes

### 19.03.2019

#### Topic Outline

##### 1.Introduction

##### 2.Image Processing for computer vision

##### 3.Camera Models and views

##### 4.Features and matching

##### 5.lightness an brightness

##### 6.image motion

##### 7.motion and tracking

##### 8.classifacation and recognition

##### 9.mechine learning

##### 10.human visions



### 21.03.2019

这里学习了怎么操作一个图片，用python写的程序。能够把一个图片转换成矩阵形式，并且能显示它的数值。



````python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#用openCV读图，并显示
img = cv2.imread('tower.jpg')
cv2.imshow('src',img)

#把图片的像素信息转换成矩阵信息
height = np.size(img, 0)
width = np.size(img, 1)
imageMatrix = np.array(img)

#打印出
print(imageMatrix[50:53,100:103])


print(height)
print(width)
print(img.dtype)
plt.plot(imageMatrix[50 , :])
plt.show()     #防止显示的图像闪退，这个图像就是折线图

````





**这里值得注意的是一个图片矩阵的每一个数值都是uint8的形式，最多只能表示255**

如果对一个图片矩阵的数值进行操作，比如叠加两个图片到一起。



````octave
dolphin = imread("dolphin.png")
bike = imread("bike.png")

%% 现在对两个图片进行叠加操作
overlay1 = dolphin/2 + bike/2

overlay2 = (dolphin+bike)/2

#这两个图像的输出是不一样的。就像之前说的usign int8，只能表示0-255的数。
````



Multiply by a scalar demo

**Octave 的数组是从1开始的！！！！！！！！！！！！**



如果想要从不同的文件夹读取文件，可以直接指定文件路径。



````octave
% Multiply by a scalar : Function
% function output = functionName(Parameter,parameter,...)
function result = scale(img,value)
  result = value.*img;  % 进行点乘，这个和matlab一样
endfunction
  
dolphin = imread("dolphin.png")
imshow(scale(dolphin,1.5))



%inspect image values
bicycle = imread("D:/Octave/IntroCV/ud810-master/course_images/bicycle.png");
dolphin = imread("D:/Octave/IntroCV/ud810-master/course_images/dolphin.png");


##figure(1)
##imshow(bicycle);
##figure(2)
##imshow(dolphin);
##disp(size(img)) ;
##
##%A slice of the image 
##img(101:103,201:203);
##plot(img(50,:))

%imshow(scale(img,1.5))
result = blend(bicycle,dolphin,0.45);
imshow(result);
````



image difference demo

````octave
dolphin = imread("dolphin.png")
bicycle = imread("bicycle.png")

diff = bicycle - dolphin;
imshow(diff);

abs_diff = abs(bicycle - dolphin);
imshow(abs_diff);

%% 这两个计算出来的数值其实是一样的。因为都是uint8，[0 - 255]。出现负数，就会变成0。

%% the right way to implement this function

%%(a-b)+(b-a) 数据类型是uint8，是可以这样进行计算的
%%
abs_diff = (dolphin - bicycle) + (bicycle - dolphin);

%% another way to do this
pkg load image

abs_diff2 = imabsdiff(dolphin,bicycle)
imshow(abs_diff2)
````



generate gaussian noise

disp 里面如果变成了逗号，就是直接在后面打印出来了。分号相当于是分行。

````octave
%% random number 1 row 100 columns 
%% randn means the mean is 0 and deviation is 1, a normal distribution
%% hist 返回两个值，一个是元素的个数（n），另一个是中心点（X）
noise = randn([1 100]);
[n,X] = hist(noise,linspace(-3,3,21));
disp([X;n]);
plot(X,n)
````

Effect of sigma on gaussian distribution

$\sigma$ 越大，产生的噪声越大，白点越多。

**Caution : **

normalization是给show image用的，在计算的时候，并不需要normalization



## 24.03.2019





















