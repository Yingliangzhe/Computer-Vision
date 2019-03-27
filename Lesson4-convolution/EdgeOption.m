%% Explore edge options
pkg load image;

%% Read an image
img = imread("D:/Octave/IntroCV/ud810-master/course_images/fall-leaves.png");
imshow(img);

%% Create a Gaussian filter
filter_size = 21;
filter_sigma = 3;
filter = fspecial("gaussian",filter_size,filter_sigma);

%% Apply it specifying an edge parameter
% "circular" "replicate" "symmetric" 
%% these parameter can be added in the imfilter function 
smoothed = imfilter(img,filter,200);
imshow(smoothed);

