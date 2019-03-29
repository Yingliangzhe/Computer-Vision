# Image processing notes

[TOC]

## 19.03.2019

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



## 21.03.2019

这里学习了怎么操作一个图片，用python写的程序。能够把一个图片转换成矩阵形式，并且能显示它的数值。

###### python 显示一个图片，并且plot出来它的信息

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



###### Multiply by a scalar demo

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



###### image difference demo

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

**randn 函数产生一个标准的（0，1）正态分布的函数。**

$\sigma​$ 越大，产生的噪声越大，白点越多。

**Caution : **

normalization是给show image用的，在计算的时候，并不需要normalization



## 24.03.2019

###### Average Assumptions:

1. The "true" value of pixels are similar to the true value of pixels neraby. 
2. The noise added to each pixel is done independentliy. 

真正的图像里面的相邻的像素值是差不多一致的，可以用平均值的方法。

还有，那些噪声对于每一个的像素块，都是独立的。



## 24.03.2019

$h(u,v) = \dfrac{1}{2\pi\sigma^2}exp(-\dfrac{u^2+v^2}{2\sigma^2})$

###### 二维高斯公式



###### Correlation Filter 相关滤波器

Correlation filter : 窗口大小是2k+1 $\times​$ 2k+1 uniform weight

$G[i,j] = \dfrac{1}{(2k+1)^2}\sum^k_{u=-k}\sum_{v=-k}^kF[i+u,j+u]​$

这是一个uniform weight(均匀连续)的相关滤波器。意思是遍历在pixel$[i,j]$ 周围所有的pixel，求它们的和。然后把这些和除以他们的相同的权重。



下面来说 non-uniform weight correlation : 

$G[i,j] = \sum^k_{u=-k}\sum_{v=-k}^kH[u,v]F[i+u,j+u]​$

这里的H就是non uniform weight

这个filter 可以是 kernel 或者mask，H矩阵就是一个有线性组合的权重矩阵。



###### Matlab function for adding gaussian noise 

fspecial : 这个函数并不能直接使用，而是需要先 pkg load image，然后才能正常使用。

h = fspecial ("gaussian",size,sigma)

这里的Gaussian表示高斯分布，size是几乘几的像素块。sigma就是公式里的variance



````octave
clear all
clc

%% Load an image
pkg load image
img = imread("D:/Octave/IntroCV/ud810-master/course_images/saturn.png");
##figure(1);
%imshow(img);

%% Add some noise
noise_sigma = 100;
noise = randn(size(img)).*noise_sigma;
noisy_img = img + noise;
##figure(2);
%imshow(noisy_img);

%% Create a Gaussian filter
filter_size = 10;
filter_sigma = 100;
filter = fspecial("gaussian",filter_size,filter_sigma);

%% Apply it to remove nosie
smoothed = imfilter(noisy_img,filter);
##figure(3);
%imshow(smoothed);

subplot(1,3,1),imshow(img)
subplot(1,3,2),imshow(noisy_img)
subplot(1,3,3),imshow(smoothed)


````



###### quiz about gaussian filter

高斯滤波中最重要的是sigma，这个定义了blur kernel的scale。而且，它改变了整体的亮度。 

相比于kernel size肯定是scale更加重要。想像一下一个2d的高斯分布，如果sigma很小，那么，就算有再大的窗口，可是没有用的。



总结一下:

在添加噪声的时候的Gaussian的 $\sigma$ 是用来作为乘数，乘以每一个像素点的，所以它会造成，每一个点的亮度增加。





#### Linearity and convolution

###### Introduction

介绍了一些关于线性系统的特征，就是加法和与常数乘。

###### Impulse function and response

###### Kernel Quiz 

假设我们有一个M $\times$ M 的原始图像和 N $\times$ N 的filter的矩阵，用这种卷积方式，会产生多少种的乘数项？答案是：

$M^2\times N^2$

对于M原始图像的每一个pixel都会进行一次filter，一次filter会有N $\times$ N 的项数。

###### Correlation and convolution

其实对比两个公式可以发现，在那个H矩阵对称的时候，这两个算法是没有区别的。不过是一个颠倒过来了。

###### Computational complexity and separability

卷积是线性的，所以对于乘法的先后顺序是没有关系的。

如果一个filter可以被分解成一个列向量*一个行向量，就可以大大缩小计算的复杂度。

原来是$W^2\times N^2$ ，现在是$2\times W\times N^2$ 。可以节省资源。 

###### Boundary Issues

如果对图像滤波的时候，滤波器超过了那个图像的边缘，这个时候怎么办？

（**首先滤波器大小肯定是奇数**）

如果用matlab的话，我们有三种选择：

- 用滤波器的角元素开始和图像进行卷积，那么就会得到一个加大版的输出图像
- 用中间的元素开始滤波可以得到一个和原图片相等大小的输出图像
- 如果滤波器完全从图像的开始地方进行滤波，就会有一个比原来小一些尺寸的输出图像。

![1553698266943](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1553698266943.png)

这个图像就表示了上面说的滤波器。

###### Methods

matlab用的方法有以下几种：
-  clip ，它把图像的周围都弄成是黑色的，然后把这些黑色的像素块和原来的图像一起进行滤波。这样得到的结果就是，输出图像会在边缘比较黑。
- wrap around方法，是对图像进行一下傅里叶变换，让它成为一个周期信号，然后进行滤波。
- 还有把edge给伸长的，但是这个带来的结果并不是很好
- 在edge处，把这些图像对称过去，然后进行滤波，最后把边缘的图像切掉。

###### linear filters
一个图象被一个impulse filter 滤波之后，是什么都不会改变的。


$$
\left\{
 \begin{matrix}
   0 & 0 & 0 \\
   0 & 1 & 0  \\
   1 & 0 & 0  \\
  \end{matrix} 
\right\}
$$
如果把两个滤波矩阵相减，像这样：

$$\left\{
 \begin{matrix}
   0 & 0 & 0 \\
   0 & 2 & 0  \\
   0 & 0 & 0  \\
  \end{matrix} 
\right\}​$$ $-​$ $$\dfrac{1}{9}\left\{
 \begin{matrix}
   1 & 1 & 1 \\
   1 & 1 & 1  \\
   1 & 1 & 1  \\
  \end{matrix} 
\right\}​$$

用这样的滤波器滤波会有什么样的效果？

答： 这个叫sharpening filter，就是图像在原有的基础上变的更锐利了。就是**黑的更黑，白的更白**。 

关于化学方法的unsharp mask。在一个底片上面放一个蜡片，然后把另外一个底片放在先前那个底片的下面，然后**打光**。 然后呈现出了sharping mask的效果。这个操作是针对两张相同的图片。两张图片就对应着那个impulse 2 的矩阵，然后那个蜡片就是那个blur。当一打光，就相当于把这两个底片叠加在一起，然后进行滤波。 

###### Median Filter

这种filter就是用一块区域的中间数来代替，原来的那个椒盐噪声的点。

![1553700429615](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1553700429615.png)

原来的矩阵中间有一个很大的值，应该就是椒盐噪声。然后，我们把整个9个数拿出来，作为一个数组，找出这9个数的中间数。用它来代替我们原来矩阵相应的位置。





## 28.03.2019

### Filter as templates

###### 1D Correlation

介绍了两个1d信号的相关性滤波的特性。如果一个信号是另一个信号截取后产生的，那么用这个信号进行滤波，在与源信号最大值重叠的时候，滤波后的信号达到最大值。



###### matlab cross correlation 

````octave
c = normxcorr2(onion,peppers)
````

当使用这个normxcorr2 的时候，这个filter实际上是从s的第一个元素开始的，并且到最后一个元素。

也就是从 3 4 5 6 7 这个顺序来的。



## 29.03.2019

在load一个package的时候，octave 和 matlab一样。function 定义都是写在最上面，然后是pkg load。



还有size函数，size(a,dim) ：

它表示了a矩阵的dim 维度的大小。

row ： dim = 1

colomn ： dim = 2

````octave
function index = find_template_1D(t,s)
  % load a package
  pkg load image;
  
  %% using the normalized cross correlation function 
  c = normxcorr2(t,s);
  %% max function can also return the index of a array element
  [maxValue rawIndex] = max(c);
  index = rawIndex - size(t,2) + 1;
  disp(index);
end

````



下面是2D的图像template match：

````octave
function [yIndex xIndex] = find_template_2D(template,img)
  % load a package
  pkg load image;
  
  %% using the normalized cross correlation function 
  c = normxcorr2(template,img);
  [yRaw xRaw] = find(c==max(c(:)));
  yIndex = yRaw - size(template,1) + 1;
  xIndex = xRaw - size(template,2) + 1;
end

````

以及相应的main 函数

````octave
%% Find template 2D

tablet = imread("D:/Octave/IntroCV/ud810-master/course_images/tablet.png");
imshow(tablet);
glyph = tablet(75:165,150:185);
glyph_height = 165-75;
glyph_width = 185-150;
imshow(glyph);

[y x] = find_template_2D(glyph,tablet);
disp([y x]);

% Plot where the template was found
colormap("gray"),imagesc(tablet);
hold on;
plot(x,y,"r+","markersize",16);
hold on;
rectangle("Position",[x y glyph_width glyph_height],"LineStyle","-","EdgeColor","r");
hold off;

````

找到相应的template，并在周围画上矩形。

![1553848467370](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1553848467370.png)

###### Non identical Template Mathing

有的时候并不是template一定要在原图片里找到，只是需要有相似的地方，就可以进行匹配。

因为这个用的是cross correlation算法，目的就是找到可能出现的最大值。所以这个不是一个绝对量。

这个查找的方法是有一个前提的，就是template基本上是给出来的。

### Edge detection : Gradients

###### Edge Detection

一个图片可以被转换成一个intensity的plot，如果看这个plot，我们会发现，有高低不同的数值。在一个像素块附近，有一个非常大的变化，可以认定这个是一个边界。



![1553851195797](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1553851195797.png)

把一个图片按照某一行进行plot，生成第二张图片。能够看到中间有一个沟壑。两边对应这非常陡的边缘。如果我们对这个进行求导，能看到两个极值。



###### what is a gradient

derivates能够返回一些导数。

然后把这个operator弄成一个滤波器。

下面是一个梯度公式：梯度是一个在x和y方向上的导数，合成的一个向量。

$\nabla f = [\dfrac{\partial f}{\partial x},\dfrac{\partial f}{\partial y}]​$

gradient direction : $\theta = tan^{-1}(\dfrac{\partial f}{\partial x}/\dfrac{\partial f}{\partial y} )$



gradient magnitude : $||\nabla f|| = \sqrt{(\dfrac{\partial f}{\partial x})^2 + (\dfrac{\partial f}{\partial y})^2}$



###### finite Differences

在图像里面我们是没有连续的导数的，只能通过这种方法进行逼近。

$\dfrac{\partial f(x,y)}{\partial x} \approx \dfrac{f(x+1,y)-f(x,y)}{1} \approx f(x+1,y)-f(x,y)$



###### Partial Derivatives of an image





![1553853732987](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1553853732987.png)



上面那个图，左面的是对x，右面的是对y

$[-1 \space 1]​$ 是对于x方向的导数filter



###### Gradient Direction

sobel operator 是 如同下面矩阵那样的。

$\left\{
 \begin{matrix}
   -1 & 0 & +1 \\
   -2 & 0 & +2  \\
   -1 & 0 & +1  \\
  \end{matrix} 
\right\}$



如果这个滤波一个左边是0（黑色），右边是1（白色）的图像。负的变成了0，正的变成了4。 所以这个滤波器，能产生的值的范围是[-4,4]。

$[-4,4] + 4 = [0,8]$

然后需要对这个滤波器进行normalization，$[0,8]/8 = [0,1]$



对于整体的scale，需要除以$4\sqrt2$ 。因为每一个的scale都是[0,4]，平方后为16，再相加32。所以就得除已这些scale。



````octave
% radient Direction
pkg load image;

%Load and convert image to double type, range[0,1] for convenience
img = double(imread("D:/Octave/IntroCV/ud810-master/course_images/octagon.png"))/255.;
imshow(img); %assumes [0,1] range for double images

% Compute x,y gradients
[gx gy] = imgradientxy(img,"sobel");
%imshow((gx+4)/8);

% obtain gradient magnitude and direction
[gmag gdir] = imgradient(gx,gy);
%imshow(gmag/(4*sqrt(2))); %use double function the edge can be brighter
%imshow((gdir + 180.0)/360);

% Find pixels with desired gradient direction
my_grad = select_gdir(gmag,gdir,1,30,60);
imshow(my_grad);
````



````octave
function result = select_gdir(gmag,gdir,mag_min,angle_low,angle_high)
  result = gdir>=angle_low & gdir<=angle_high & gmag >= mag_min;
end
````

用来打印出不同角度的边界图形。



### Edge Detection using 2D Operator

 

###### Effect of Sigma on Derivatives

对于一个真实场景，图片都是有高频噪声的。如果贸然进行梯度法求导，那个再数值上根本无法看出来有一个极值。有的只是高频噪声。














