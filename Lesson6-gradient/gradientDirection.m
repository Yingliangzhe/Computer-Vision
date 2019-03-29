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