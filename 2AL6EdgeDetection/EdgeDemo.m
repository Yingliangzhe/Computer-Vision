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

