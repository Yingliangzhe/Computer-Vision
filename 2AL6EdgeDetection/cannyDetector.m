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