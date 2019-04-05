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

用一个Gaussian filter来对一个图片进行滤波，$\sigma$ 的值越大，对于 一个图片的模糊程度影响越大。

![1554118584413](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554118584413.png)



从这个图片就能看出来，如果$\sigma$ 越大，边界的detection就越不明显。



###### Canny Edge Operator

步骤：

1. 先用高斯对图像进行滤波
2. 然后找到*梯度* 的幅值和方向
3. 找到non maximal suppression，把多个pixel的ridge，细化到单个的pixel
4. 连接，并设置阈值。定义两个阈值，大的用来开始edge curves，小的用来使它们连接起来。

![1554119726306](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554119726306.png)



这个图片里，框里面的白线是一条edge。如果截取这个白线，然后把这个给plot一下。从plot里面能看出来，那个edge的地方是一个上升期。如果求导，就会有处有极值。同时设置一个threshold，在这个阈值以上的地方就被定义为edge。



**最重要的canny threshold hysteresis**

1. 设置一个high threshold用来检测强像素点
2. 把强像素点连接起来，使这个形成一个强边界
3. 然后弄一个low threshold，检测那些可能是边界的像素点
4. 把强边界延伸开来，去跟随那些若边界。



````octave
% for your eyes only
pkg load image;

frizzy = imread("D:/Octave/IntroCV/ud810-master/course_images/frizzy.png");
froomer = imread("D:/Octave/IntroCV/ud810-master/course_images/froomer.png");

imshow(frizzy);
imshow(froomer);

%% find edges in frizzy and frommer images
frizzy_gray = rgb2gray(frizzy);
froomer_gray = rgb2gray(froomer);

frizzy_edges = edge(frizzy_gray,"canny");
froomer_edges = edge(froomer_gray,"canny");

imshow(frizzy_edges);
imshow(froomer_edges);

%% display the commen edge pixels
imshow(frizzy_edges & froomer_edges);
````

因为这里的图像就是0 和 1了，所以可以直接用&运算符。

这样就能显示两个共同的pixel了。



###### Single 2D Edge Detection Filter

之前已经说过用1D 的 Gaussian filter对信号进行滤波。如果是2D的情况怎么办？一个函数的二阶导数，应该用哪个导数。x对x，y对y，还是x对y？



最后我们决定用Laplace operator 来对一个函数进行求导。

$\nabla^2 h= \dfrac{\partial^2 f}{\partial x^2}+\dfrac{\partial^2 f}{\partial y^2}​$

用来求一个场中的散度。



###### Edge demo

下面这个例子是用来展示三种不同的对于找边界的方法。

其中有一个关于矩阵的操作。

17行，把从第二个到最后一个的像素点 赋给 从第一个到end - 1的像素点。这样就是图片左移。前面的冒号代表所有的row 。

````octave
pkg load image;

lena = imread("D:/Octave/IntroCV/ud810-master/course_images/lena.png");
%figure,imshow(lena),title("Original image, color");

%% Convert to monochrome using rgb2gray
lenaMono = rgb2gray(lena);
figure,imshow(lenaMono),title("Original image, monochrome");

%% make a blurred version
h = fspecial("gaussian",[11 11],4);
%figure,surf(h);
lenaSmooth = imfilter(lenaMono,h);

%% Methode 1: shift left and right, and show diff image
lenaL = lenaSmooth;
lenaL(:,[1:(end - 1)]) = lenaL(:,[2:end]);
lenaR = lenaSmooth;
lenaR(:,[2:end]) = lenaR(:,[1:(end - 1)]);
lenaDiff = double(lenaR) - double(lenaL);
figure,imshow(lenaDiff,[]),title("Difference betwwen right and left shifted images");

%% Methode 2: canny edge detector
cannyEdges = edge(lenaSmooth,"Canny");
figure,imshow(cannyEdges),title("Edges of smoothed image");

%% Methode 3 : laplacian of Gaussian
logEdges = edge(lenaMono,"log");
figure,imshow(logEdges),title("Laplacian of Gaussian");


````





### Hough transform : Lines

###### Line fitting

首先说一下，为什么我们不能继续用这个edge detector。

![1554127363022](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554127363022.png)

看这个图片就知道，如果用了edge detector，有一些图片上的细节，其实不是我们想要的line，但是也会被当成边界。这就大大提高了计算消耗。

而且，有一些line的部分是没法检测出来的。

有一些是edge其实不是真正的edge，可能是因为noise的作用。

于是为了解决这个问题，我们就用Hough transform算法。





## 02.04.2019



###### Hough Space

###### Basic Hough transform algorithm

在一个H（d,$\theta$） 的矩阵里。一个二维矩阵。

1. 初始化这个矩阵 = 0
2. 对于每一个在原图像的**edge** 点 ，对于每一个$\theta$ 从0到180 遍历。计算一个d，然后，让H(d,$\theta$) + 1 作为一个vote值。**这里一定注意！！！！！！是边界点。**
3.   遍历完之后，对这个vote 矩阵找最大值。最大值是一个关于$ \d_0 ​$ 和 $\theta_0​$  的函数。
4. 那么检测出来的线，就是 $d_0 = xcos\theta_0+ysin\theta_0$ 

###### Complexity of the hough transform

时间复杂度：就是检验出的line，或者edge point的数量。



###### Hough example

![1554196792903](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554196792903.png)



在右面的图像里，d和$\theta​$ 空间。每一个点都是一个sinus的函数，那么这些函数的交点，就是一条直线。



###### Hough Demo

````octave
pkg load image;

%%load image, convert to grayscale and apply canny operator
img = imread("D:/Octave/IntroCV/ud810-master/course_images/shapes.png");
grays = rgb2gray(img);
edges = edge(grays,"canny");

figure,imshow(img),title("Original image");
figure,imshow(grays),title("gray image");
figure,imshow(edges),title("edge image");

%% Apply hough transform to find candidate lines
accum = houghtf(edges);
figure,imagesc(accum),xlabel("theta"),ylabel("d"),title("Hough accumulator");
````

这个程序用来展示出那个H矩阵。在这个图像里，所有的点都是作为正弦函数表现出来的。

而这些正弦函数的交点，就是原edge图像的line。

还有，matlab的houghtf()的返回值，只有一个H矩阵。



![1554201390784](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554201390784.png)



hough函数还可以有不同的parameter，比如threshold和HNoodSize。

threshold是用来界定peak值的最小值。

HNoodSize是一个二维矩阵，指的是围绕峰值的邻域大小。当在这个邻域里检测出峰值后，就把整个邻域设置为0。

所以如果这个邻域值设置的过大，比如是infinity，可能在原图片里只能检测出一个line。

如果过于小，可能有很多其实不是直线的线，也会被检测成直线。



### Hough tranformation  : circles

**对于给出半径的圆。**

如同下面这个图：

$(x_i - a)^2+(y_i -b)^2 = r^2​$

在x y空间是一个圆的表达式。如果把这个表达式在a b 平面展示出来是什么样的？

就是把公式里面的东西给换一下位置，$(a-x_i )^2+(b- y_i )^2 = r^2$

变成了一个以$x_i 和 y_i$ 为圆心的一个圆。

在圆上找三个点，在ab空间就会映射出来三个圆，而这三个圆一定会有一个交点。这个交点，就是xy平面圆的圆心。

![1554206884003](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554206884003.png)



###### Hough Transformation for Circles

对于不知道半径的圆。

那么在Hough空间就会出现几个圆锥，这些圆锥的表面的交点。注意是表面的，不是里面的。

这样我们的voting 矩阵就是三维的。这就有点麻烦。

![1554208149433](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554208149433.png)



但是在这个图片中用gradient，虽然半径不知道，但是这个圆心肯定是在这条切线的法向量方向上的。

最终，还是会有一个交点在 $a -b -r​$ 空间上。

![1554208276978](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554208276978.png)





````c
For every edge pixel(x,y):
	For each possible radius value r:
		For each possible gradient direction theta:
			a = x-rcos(theta)
			b = y+rsin(theta)
			H[a,b,r]+=1
		endFor
	endFor
endFor
````



### Generalized Hough Transformation



![1554211091301](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554211091301.png)



这个表就是一个按照$\theta$ 作为index的 r vector。因为新的图形可能是不规则的。可能一个$\theta$ 对应了不同的r 向量。



这里有必要说明一下这generalized Hough Transformation的作用。在之前我们做的hough transformation都是已知的一个图形的几何方程。就是在图片中去找相应的line和circle。但是对于没有几何方程的，形状不一的物体要怎么在图片里面找到呢？就可以用一般性的hough 变换。

我们先对一个模板进行分析，就是建立一个R Table。在上面的图里已经有怎么建立这个table了。就不多说。

我觉得这个table有点像描点作图，来绘制函数图像。这个table包含了整个模板所有边界点pixel的信息。



如果这个图片的object的旋转是知道的。那么就是正常的hough transformation 

![1554213949209](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554213949209.png)



## 03.04.2019

### Fourier Transformation

###### Basic set

就是我们平时说的子空间。这个是一个空间的子空间，并且能张成一个空间。

这个空间对于加法和数乘封闭。



一个N$\times $N的图像，这些图像上的点都可以通过向量的点来表示。

下面的截图，有地方错了，x方向向下，y方向向右。那个应该是01。

但是这个空间选定的不是很好。因为每一个元素都代表了一个pixel。

![1554281210073](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554281210073.png)



###### Fourier

为了能够更好的表示一个图片的basic set。我们用这个fourier 空间。它按照每个方向上的变化快慢来区分。

x方向上的越往下变化越快，y方向上也是一样。颜色什么的可以用sin和cos函数来表示。

把这些set给plot一下，应该就能看出在一个2d的平面，这些是什么样的了。其实就是一些正弦曲线，那些黑白的分界线不是那么明显，恰恰说明了这个问题。越往外，这些信号的频率越高。

![1554281763625](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554281763625.png)



想要得到一个方波，放上面加正弦波就好了，注意这个正弦波的幅值和频率。赋值是随着频率的增大而减小的。



![1554282704248](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554282704248.png)



###### Fourier Transformation

傅里叶变换可以在频域里表示幅值和相位。

$F(\omega) = R(\omega)+iI(\omega)$

$A = \plusmn\sqrt{R(\omega)^2+I(\omega)^2}​$

$\varphi = tan^{-1}\dfrac{I(\omega)}{R(\omega)}$



###### Computing Fourier Transform

这里可以先说一下，为什么两个频率不同的正弦或者余弦波相乘，再在一个周期下进行积分，得到的结果会是0？

考虑以下函数: 

$f(t) = sin(w_1 t+\varphi_1)sin(w_2t+\varphi_2)$

先把这个函数用积化和差进行分解：

$f (t) = -\dfrac{1}{2}(cos(w_1t+\varphi_1+w_2t+\varphi_2)+cos(w_1t+\varphi_1-w_2t-\varphi_2))$

​        $= -\dfrac{1}{2}(cos[(w_1+w_2)t+(\varphi_1+\varphi_2)]+cos[(w_1-w_2)t+(\varphi_1-\varphi_2)])$

我们对这个函数在$[-\infin,+\infin]​$ 区间对于$f(t) ​$ 进行积分， 会得到这样的结果：

如果$w_1 = w_2$ ，那么前面那项cos函数，在一个周期内积分会是0。后面那项，会有一个常数项。在积分的时候，直接就可以乘以积分区间。

如果$w_1 \neq w_2 $ ，那么两项全都是cos函数，两个函数在一个周期的积分下，都会是0。





## 04.04.2019

- [ ] 把这个在图像处理中的傅里叶变换搞懂
- [ ] 所有对于傅里叶变换原理的理解希望都在今天解决

对于傅里叶变换的相位谱，我们可以把每个波的波峰给标记出来。然后把这些波峰点给投影到时间轴上，那么，这个点在时间轴上到零点（频率轴）的距离，就是时间差（用时间差可以计算相位 $相位=（时间差\div周期）\times 2\pi$）。



于是就有了相位谱。至于相位谱的值域$[-\pi,\pi]$ 。因为正弦函数是一个以$2\pi$的整数倍为周期的函数，这个区间已经可以涵盖所有的相位了。



![1554374898597](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554374898597.png)



###### Fourier Transform More Formally

傅里叶公式：

$F(u) = \int_{-\infin}^{\infin}f(x)e^{-i2\pi ux}dx$ ， $u$ 代表频率

$f(x)$ 是那个原函数。$F(u)$ 是在频域里的表现形式。



对于二维的图像（二维的傅里叶级数），我们有下面的图片：

![1554384155314](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554384155314.png)



![1554384177814](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554384177814.png)

## 05.04.2019

###### Fourier Transform -> Fourier Series

###### 2D Fourier Transform

二维连续傅里叶变换：

$F(u,v) = \dfrac{1}{2}\int_{-\infin}^{\infin}\int_{-\infin}^{\infin}f(x,y)e^{-i2\pi (ux+vy)}dxdy$

二维离散傅里叶变换：

$F(k_x,k_y) = \dfrac1N\sum_{x=0}^{x=N-1}\sum_{y=0}^{y=N-1}f(x,y)e^{-i\dfrac{2\pi(k_xx+k_yy)}{N}}$



一个二维图像的base。图像中的所有值都可以用这些基的线性组合来组成。下面画红线的区域就是这个基。

![1554448988621](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554448988621.png)





如同下面的图，把一个x方向的正弦波进行傅里叶变换，可以看到在频域里的图像就是两个点。向x方向传播的就是沿u方向的两个点。如果波沿着y方向传播，频谱图就是沿着v方向的两个点。

![1554449161831](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554449161831.png)





下面是把两个方向的波给合成起来。就是两个斜着的点。

![1554449198515](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554449198515.png)



由于傅里叶变换的线性的特征，在频谱上面表现出来的就是叠加。

![1554449236292](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554449236292.png)



###### Examples

一般自然图片的能量谱看起来都差不多，如果想把如片重新合成，需要的信息是从相位谱来的。



看下面的图2，如果我们把周围的频谱去掉，只保留中间的高频部分，相当于进行了一个滤波。

图3，我们把原图片频谱中的**低频部分**去掉，然后进行傅里叶逆变换。可以看到，原图片就剩下了边界edge。

![1554450567683](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554450567683.png)





如果把原图片的能量谱在每个频率下的能量加强，可以看出，图片变的更sharp。

对于墙和方块的图片，把两个图片进行傅里叶变换，有的频谱只是关于水平和垂直方向的。

![1554450840275](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554450840275.png)



### Convolution in frequency domain

###### Fourier Transform and convolution

想象一下，有两个函数要进行卷积（convolution）：

$g = f*h$

对这个g进行傅里叶变换：$G(u) = \int_{-\infin}^{\infin}g(x)e^{-i2\pi ux}dx​$

​						       $ = \int_{-\infin}^{\infin}\int_{-\infin}^{\infin}f(\tau)h(x-\tau)e^{-i2\pi ux}d \tau dx​$

​						       $ = \int_{-\infin}^{\infin}\int_{-\infin}^{\infin}[f(\tau)e^{-i2\pi u\tau}d \tau] [h(x-\tau)e^{-i2\pi u(x-\tau)}]dx$

​					       	$ =  \underbrace{\int_{-\infin}^{\infin}[f(\tau)e^{-i2\pi u\tau}d \tau]}_{F(u)}\underbrace{\int_{-\infin}^{\infin} [h(x^{'})e^{-i2\pi ux^{'}}]dx^{'}}_{H(u)}$

​							$ = F(u)H(u)​$

通过这一系列的推导就能看出，如果把两个函数给进行卷积，然后再进行傅里叶变换，那么，得到的就相当于把两个函数先分别傅里叶变换，再相乘。



![1554453387781](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554453387781.png)



###### smoothing and bluring

就像之前介绍过的一样，如果想对一个有噪声的信号进行滤波，我们可以用一个高斯滤波器去和原图像卷积。

如果把这个高斯滤波器给进行傅里叶变换，还是会得到一个在频域的高斯分布。而且这个频域内的高斯分布，在图像上呈现的宽会窄一些。因为$\sigma $ 在分号下面。

这个性质在频域滤波的表现形式就是，会保留中间频率的特性，而其他地方的特性会被滤掉。趋向于0。



![1554453648623](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554453648623.png)



###### Low and High Pass Filtering

下面这个图，表示的是高通和低通滤波器。中间的图片表示的是两个滤波器在频域的呈现。由于中间的坐标表示的是低频状态。外部的表示高频。所以一个只保留中间图像的是低通滤波器，而一个保留外部的是高通滤波器。



![1554454686106](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554454686106.png)



###### Fourier Pairs



之前我们讲过，如果用一个方块的filter去滤波，会得到像右边那个非常丑的图。本来不应该有白色的地方有白色。因为高频部分有像sine wave的值。

但是下面那个高斯滤波器就显得非常好，因为在高频的部分是没有值的。

![1554455629810](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554455629810.png)



### Aliasing

###### Sampling and Reconstruction

###### Aliasing

想象一下这个是飞机螺旋桨。这里旋转的每一下，都好像是顺时针旋转90度。那里有一个点，还方便大家看到底哪个是哪个。所以大家能够清楚地看出来这个轮子是朝哪个方向转的。但是一旦没有了那个点，我们可能觉得这个轮子会朝反方向转。



![1554462326214](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554462326214.png)



看下面这张图，我们把一个chirp在2D 图像上显示出来。前面的正弦波都是正常的显示，黑白黑白。越到后面，频率越高，所以条条越窄。但是到了这个最后面都是黑色的？？？WTH？

就是说，没有足够的sampling frequency来保证信号被采集到。

![1554466394362](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554466394362.png)



###### Antialiasing

如何避免产生Aliasing？

有以下几个点：

1. 增加我们的像素，如果原来有3600k 像素，我们增加到36Mega Pixel。
   - 但是这样并不是可以一直见效的。
2. 可以让原图片减少一些晃动
   - 高频的一些噪声我们不要了
   - 可能会丢失一些信息
   - 但是这也好过Aliasing



###### Impulse Train and Bed of Nails

$comb_M[x] = \sum_{k=-\infin}^{\infin}[x-kM] $   这里的$M$ 表示在轴上的间隔。

这个就是一个set，由impulse组成的set。

一个时域里由impulse组成的set，在频域里也是一个由impulse 组成的set。

注意傅里叶变换有一个特质，就是：如果x的取值发生了变化（变小），那么，在频域里的取值会变大。

这里的beds of nails 应该是钉板的意思。把2D的impulse给plot出来，就是像一个钉板。



###### Sampling Low Frequency Signal

**时域的卷积，就是频域的乘法**

**时域的乘法，就是频域的卷积**

看下面这个图片。图里面的$f(x)$ 是一个比较好的低频信号。把这个信号进行傅里叶变换，会得到右面的图形。

我们对原信号进行采样，再进行傅里叶变换。这次我们得到的再频域里的信号就是一个周期信号。

由于再采样的时候，我们的时间间隔M足够小，那么在频域里面的信号的间隔就会比较大。如果进行了变换后的信号的最大信号，没有超过$\dfrac{1}{M}$ ，信号就可以原原本本地被还原回去。

![1554470547116](C:\Users\jialiang.yin\AppData\Roaming\Typora\typora-user-images\1554470547116.png)















































































