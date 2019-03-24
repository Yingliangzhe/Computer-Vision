%inspect image values
bicycle = imread("D:/Octave/IntroCV/ud810-master/course_images/bicycle.png");
dolphin = imread("D:/Octave/IntroCV/ud810-master/course_images/dolphin.png");
saturn = imread("D:/Octave/IntroCV/ud810-master/course_images/saturn.png");


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
##result = blend(bicycle,dolphin,0.45);
##imshow(result);

##abs_diff = (dolphin - bicycle) + (bicycle - dolphin);
##imshow(abs_diff)

%% random number 1 row 100 columns 
##noise1 = randn([1 100]);
##noise2 = randn([1 100]);
##[n1,X1] = hist(noise1,linspace(-3,3,30));
##[n2,X2] = hist(noise2,linspace(-3,3,30));
##mesh(X1,X2,4-X1-X2)
##disp([X;n]);
##plot(X,n)
disp(size(saturn))
figure(1)
imshow(saturn)

noise = randn(size(saturn)).*25;
output = saturn + noise;
figure(2)
imshow(output)
















