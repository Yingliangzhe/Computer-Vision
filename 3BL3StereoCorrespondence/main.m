pkg load image;

%% load images
left = imread("D:/Octave/IntroCV/ud810-master/course_images/flowers-left.png");
right = imread("D:/Octave/IntroCV/ud810-master/course_images/flowers-right.png");

%% Convert to grayscale, double,[0,1] range for easier computation
left_gray = double(rgb2gray(left))/255.0;
right_gray = double(rgb2gray(right))/255.0;

%% Define image patch location(topleft[row coll]) and size
patch_loc = [120 170];
patch_size = [100 100];

%% Extract patch (from left image)
patch_left = left_gray(patch_loc(1):patch_loc(1)+patch_size(1) - 1,
   patch_loc(2):patch_loc(2) + patch_size(2) - 1); 
figure,imshow(patch_left);   

%% Extract strip (from right image)
strip_right = right_gray(patch_loc(1):(patch_loc(1) + patch_size(1) - 1),:);
figure,imshow(strip_right);
 
%% look for the patch in the strip 
%% and report the best position(column index of topleft corner)
best_x = find_best_match(patch_left,strip_right);
disp(best_x);
patch_right = right_gray(patch_loc(1):(patch_loc(1) + patch_size(1)-1),
   best_x:(best_x + patch_size(2)-1));
figure,imshow(patch_right);