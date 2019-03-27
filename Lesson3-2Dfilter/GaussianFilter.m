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

