%% Apply a Median filter
pkg load image

%% Read an image
img = imread("D:/Octave/IntroCV/ud810-master/course_images/moon.png");
imshow(img);

%% Add salt and pepper noise
noisy_img = imnoise(img,"salt & pepper",0.02);
imshow(noisy_img);

%% Apply a median filter
%% medfilt2 means median filter for 2D
median_filtered = medfilt2(noisy_img);
imshow(median_filtered);

%% Apply a Gaussian filter
image_gaussian = imfilter(img,filter,0);
imshow(image_gaussian);