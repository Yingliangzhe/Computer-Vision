%% Find template 1D


%s = [-1 0 0 1 1 1 0 -1 -1 0 1 0 0 -1];
%t = [1 1 0];

%disp("Signal:"),disp([1:size(s,2);s]);
%disp("Template:"),disp([1:size(t,2);t]);

%index = find_template_1D(t,s);
%disp("Index:"),disp(index)

%m = add(1,2);
%disp(m);

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


