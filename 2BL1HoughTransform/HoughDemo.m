pkg load image;

%%load image, convert to grayscale and apply canny operator
img = imread("D:/Octave/IntroCV/ud810-master/course_images/shapes.png");
grays = rgb2gray(img);
edges = edge(grays,"canny");

figure,imshow(img),title("Original image");
figure,imshow(grays),title("gray image");
figure,imshow(edges),title("edge image");

%% Apply hough transform to find candidate lines
[accum, theta, rho] = hough(edges);
figure,imagesc(accum),title("Hough accumulator");

%% find peaks in the Hough accmulator matrix
peaks = houghpeaks(accum,100);
hold on; plot(theta(peaks(:,2)),rho(peaks(:,1)),"rs");hold off;

