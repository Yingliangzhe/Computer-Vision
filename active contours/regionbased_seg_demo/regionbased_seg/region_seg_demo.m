% Demo of "Region Based Active Contours"
%
% Example:
% seg_demo
%
% Coded by: Shawn Lankton (www.shawnlankton.com)
pkg load image;

I = imread('affine_0.bmp');  %-- load the image
disp(size(I))
m = zeros(size(I,1),size(I,2));          %-- create initial mask
m(400:1700,900:2100) = 1; % m(111:222,123:234) = 1;  for airplane
% octave?????? ????????????

I = imresize(I,.5);  %-- make image smaller 
m = imresize(m,.5);  %     for fast computation

subplot(2,2,1); imshow(I); title('Input Image');
subplot(2,2,2); imshow(m); title('Initialization');
subplot(2,2,3); title('Segmentation');

seg = region_seg(I, m, 1200); %-- Run segmentation ite_max = 250

subplot(2,2,4); imshow(seg); title('Global Region-Based Segmentation');


