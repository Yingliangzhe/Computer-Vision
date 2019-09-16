%This script file performs the Multi-scale edge detection using Gaussian and
%Laplacian Pyramids

% Author 	: Shan
% Email		: smilingsantanu@gmail.com  :)
% Date		: October 2012
% Version	: 1.0 

clear; clc; close all;

%Initials Tests of the images 
%Read a test image
OriginalImg = imread('CARTOON.jpg');
%Disply the original Image
imshow(OriginalImg)

NoOfImages=5;
%List the images in a cell for future use
ImageNames={'CARTOON.jpg', 'flowergray.jpg', 'kitty.jpg', 'polarcities.jpg', ...
        'text.jpg'}
ImageNamesConv1={'CARTOON1.jpg', 'flowergray1.jpg', 'kitty1.jpg', 'polarcities1.jpg', ...
        'text1.jpg'}
ImageNamesConv2={'CARTOON2.jpg', 'flowergray2.jpg', 'kitty2.jpg', 'polarcities2.jpg', ...
        'text2.jpg'}    
%Read all the images 
for i=1: NoOfImages
    OriginalImages{i}=imread(ImageNames{i});
end

%show all images
for i=1: NoOfImages
    figure(i)
    imshow(OriginalImages{i});
end

% pause and closee all the images
pause (2)
close all;
%Preparation / check on the images for the given Kernels
% List of keranls K = { h1, h2,...}
NoOfKernels=2;
Kernels={ [ones(2,2)*1/4]; [ones(3,3)*1/9]};

%Perform Convolution on the images 
for i=1: NoOfKernels    
    for j=1 : NoOfImages 


        Kernel= Kernels{i};
        InputImg=OriginalImages{j};
        ConvImg = convolve(InputImg, Kernel); 
        ConvImages{i,j} = ConvImg;
        
        ConvImagesSizes{i,j}=size(ConvImg);
%         ConvImgRGB(1:size(ConvImg,1), 1:size(ConvImg,2), 1)=ConvImg;
%         ConvImgRGB(1:size(ConvImg,1), 1:size(ConvImg,2), 2)=zeros(size(ConvImg));
%         ConvImgRGB(1:size(ConvImg,1), 1:size(ConvImg,2), 3)=zeros(size(ConvImg));
%         ConvImagesRGB{i,j}=ConvImgRGB;
    end
end

%Display/show and store the convolved images
cmap = colormap;

prntsz=7.5;
for i=1: NoOfKernels
    for j=1: NoOfImages
        f=figure, imshow(ConvImages{i,j}, 'DisplayRange', [0 255]);
        %imtool(ConvImages{i,j}, 'DisplayRange', [0 255]);
        pause(2)
        %Store the output images to file in jpeg format
        switch(i)
            case 1
                cmap = colormap('gray');
                imwrite(ConvImages{i,j},cmap, ImageNamesConv1{j}, 'jpeg');
                
                
                %print(['p_',ImageNamesConv1{j}], '-djpeg')
                               
                set(f,'units','centimeters','position',[1 1 prntsz prntsz]) % set the screen size and position
                set(f,'paperunits','centimeters','paperposition',[1 1 prntsz prntsz]) % set size and position for printing
                set(gca,'units','normalized','position',[0 0 1 1]) % make sure axis fills entire figure
                print(f, '-r300','-djpeg',['p_',ImageNamesConv1{j}])
            case 2
                cmap = colormap('gray');
                imwrite(ConvImages{i,j},cmap, ImageNamesConv2{j}, 'jpeg');
                print(['p_',ImageNamesConv2{j}], '-djpeg')
                set(f,'units','centimeters','position',[1 1 prntsz prntsz]) % set the screen size and position
                set(f,'paperunits','centimeters','paperposition',[1 1 prntsz prntsz]) % set size and position for printing
                set(gca,'units','normalized','position',[0 0 1 1]) % make sure axis fills entire figure
                print(f, '-r80','-djpeg',['p_',ImageNamesConv2{j}])

        end
    end
end

close all;

%==========================================================================
%Assignment 1: Low-pass Filtering
% Write a program to generate an 8 level Gaussian pyramid using convolution. Use two different
% approaches to generate the pyramid. Generate each level of the pyramid by applying a 2x2 box
% filter to the image in the immediately preceding level.

%Inputs: Gray Image, Box Filter (2x2)
%Output: Filtered output at various step levels

%gaussian pyramid is computed by the impyramid 
% The steps include 

%==========================================================================
% Step1: Prepare the inputs and the kernel i.e box filter

index=1;
for j=1:NoOfImages  % do the same for five images
close all;    
I0=double(OriginalImages{j}); %input image
BoxKernel=1/4*ones(2);
figure, imshow(uint8(I0));
cmap = colormap('gray');
imwrite(I0,cmap, ['I0_',ImageNamesConv1{j}], 'jpeg');
               
%Level 1 Processing
I01=convolve(I0,BoxKernel,'same');
I1=imresize(I01,0.5, 'Method', 'bilinear'); %GP layer 1     
H1=I0-I01;            %LP layer 1
size(I1)
figure, imshow(uint8(I1));
cmap = colormap('gray');
imwrite(I1,cmap, ['I1_',ImageNamesConv1{j}], 'jpeg');

%Level 2 Processing
I02=convolve(I1,BoxKernel,'same');
I2=imresize(I02,0.5, 'Method', 'bilinear'); %GP layer 2     
H2=I1-I02;            %LP layer 2
size(I2)
figure, imshow(uint8(I2));
cmap = colormap('gray');
imwrite(I2,cmap, ['I2_',ImageNamesConv1{j}], 'jpeg');

%Level 3 Processing
I03=convolve(I2,BoxKernel,'same');
I3=imresize(I03,0.5, 'Method', 'bilinear'); %GP layer 3     
H3=I2-I03;            %LP layer 3
size(I3)
figure, imshow(uint8(I3));
cmap = colormap('gray');
imwrite(I3,cmap, ['I3_',ImageNamesConv1{j}], 'jpeg');

%Level 4 Processing
I04=convolve(I3,BoxKernel,'same');
I4=imresize(I04,0.5, 'Method', 'bilinear'); %GP layer 4     
H4=I3-I04;            %LP layer 4
size(I4)
figure, imshow(uint8(I4));
cmap = colormap('gray');
imwrite(I4,cmap, ['I4_',ImageNamesConv1{j}], 'jpeg');

%Level 5 Processing
I05=convolve(I4,BoxKernel,'same');
I5=imresize(I05,0.5, 'Method', 'bilinear'); %GP layer 5     
H5=I4-I05;            %LP layer 5
size(I5)
figure, imshow(uint8(I5));
cmap = colormap('gray');
imwrite(I5,cmap, ['I5_',ImageNamesConv1{j}], 'jpeg');

%Level 6 Processing
I06=convolve(I5,BoxKernel,'same');
I6=imresize(I06,0.5, 'Method', 'bilinear'); %GP layer 6     
H6=I5-I06;            %LP layer 6
size(I6)
figure, imshow(uint8(I6));
cmap = colormap('gray');
imwrite(I6,cmap, ['I6_',ImageNamesConv1{j}], 'jpeg');

%Level 7 Processing
I07=convolve(I6,BoxKernel,'same');
I7=imresize(I07,0.5, 'Method', 'bilinear'); %GP layer 7     
H7=I6-I07;            %LP layer 7
size(I7)
figure, imshow(uint8(I7));
cmap = colormap('gray');
imwrite(I7,cmap, ['I7_',ImageNamesConv1{j}], 'jpeg');

%Level 8 Processing
I08=convolve(I7,BoxKernel,'same');
I8=imresize(I08,0.5, 'Method', 'bilinear'); %GP layer 7     
H8=I7-I08;            %LP layer 7
size(I8)
figure, imshow(uint8(I8));
cmap = colormap('gray');
imwrite(I8,cmap, ['I8_',ImageNamesConv1{j}], 'jpeg');




for i=1:8
    figure, imshow(uint8(eval(['H', num2str(i)]))) 
end
pause(10);
close all;

%==========================================================================
%Assigment 2: Band-pass Filtering
% Write a program to generate 7 levels of the Laplacian pyramid by subtracting the consecutive
% levels of the Gaussian pyramid. Use bilinear interpolation to upsample the image of lower size so
% that the two images used during subtraction have the same size.

%laplacian computation : the laplacian is a bandpass filetred output . It
%is the differnce between the two consecutive levels of Gaussian Pyramids

%for j=1:NoOfImages 
%Thsi method is bu upscaling btween consecutive levels such that we get a
%tappered pyramid of Laplacian (differnt size on each layer)

I1up=imresize(I1,2, 'Method', 'bilinear');
L0=I0-I1up;
size(L0)
figure, imshow(uint8(L0));
cmap = colormap('gray');
imwrite(L0,cmap, ['L0_', ImageNames{j}], 'jpeg');

I2up=imresize(I2,2, 'Method', 'bilinear');
L1=I1-I2up;
size(L1)
figure, imshow(uint8(L1));
cmap = colormap('gray');
imwrite(L1,cmap, ['L1_', ImageNames{j}], 'jpeg');

I3up=imresize(I3,2, 'Method', 'bilinear');
L2=I2-I3up;
size(L2)
figure, imshow(uint8(L2));
cmap = colormap('gray');
imwrite(L2,cmap, ['L2_', ImageNames{j}], 'jpeg');

I4up=imresize(I4,2, 'Method', 'bilinear');
L3=I3-I4up;
size(L3)
figure, imshow(uint8(L3));
cmap = colormap('gray');
imwrite(L3,cmap, ['L3_', ImageNames{j}], 'jpeg');

I5up=imresize(I5,2, 'Method', 'bilinear');
L4=I4-I5up;
size(L4)
figure, imshow(uint8(L4));
cmap = colormap('gray');
imwrite(L4,cmap, ['L4_', ImageNames{j}], 'jpeg');

I6up=imresize(I6,2, 'Method', 'bilinear');
L5=I5-I6up;
size(L5)
figure, imshow(uint8(L5));
cmap = colormap('gray');
imwrite(L5,cmap, ['L5_', ImageNames{j}], 'jpeg');

I7up=imresize(I7,2, 'Method', 'bilinear');
L6=I6-I7up;
size(L6)
figure, imshow(uint8(L6));
cmap = colormap('gray');
imwrite(L6,cmap, ['L6_', ImageNames{j}], 'jpeg');


I8up=imresize(I8,2, 'Method', 'bilinear')
L7=I7-I8up;
size(L7)
figure, imshow(uint8(L7));
cmap = colormap('gray');
imwrite(L7,cmap, ['L7_', ImageNames{j}], 'jpeg');

close all;


%Gaussian Pyramid with equal image sizes at each layer
G0=I0;
figure, imshow(uint8(G0));
cmap = colormap('gray');
imwrite(G0,cmap, ['G0_', ImageNames{j}], 'jpeg');

G1=imresize(I1,2, 'Method', 'bilinear');
figure, imshow(uint8(G1));
cmap = colormap('gray');
imwrite(G1,cmap, ['G1_', ImageNames{j}], 'jpeg');

G2=imresize(I2,4, 'Method', 'bilinear');
figure, imshow(uint8(G2));
cmap = colormap('gray');
imwrite(G2,cmap, ['G2_', ImageNames{j}], 'jpeg');

G3=imresize(I3,8, 'Method', 'bilinear');
figure, imshow(uint8(G3));
cmap = colormap('gray');
imwrite(G3,cmap, ['G3_', ImageNames{j}], 'jpeg');

G4=imresize(I4,16, 'Method', 'bilinear');
figure, imshow(uint8(G4));
cmap = colormap('gray');
imwrite(G4,cmap, ['G4_', ImageNames{j}], 'jpeg');

G5=imresize(I5,32, 'Method', 'bilinear');
figure, imshow(uint8(G5));
cmap = colormap('gray');
imwrite(G5,cmap, ['G5_', ImageNames{j}], 'jpeg');

G6=imresize(I6,64, 'Method', 'bilinear');
figure, imshow(uint8(G6));
cmap = colormap('gray');
imwrite(G6,cmap, ['G6_', ImageNames{j}], 'jpeg');

G7=imresize(I7,128, 'Method', 'bilinear');
figure, imshow(uint8(G7));
cmap = colormap('gray');
imwrite(G7,cmap, ['G7_', ImageNames{j}], 'jpeg');

%Laplacian computation with equal size in images across layers
L0eq=G0-G1;
L1eq=G1-G2;
L2eq=G2-G3;
L3eq=G3-G4;
L4eq=G4-G5;
L5eq=G5-G6;
L6eq=G6-G7;

    for i=0:6

        figure, imshow(uint8(eval(['L', num2str(i),'eq'])))
        cmap = colormap('gray');
        imwrite(eval(['L', num2str(i),'eq']),cmap, ['L', num2str(i),'eq_', ImageNames{j}], 'jpeg');

    end

close all;


%Assignment 3: Multi-Scale Edge Detection
% Step 1: Generate the second order derivative images at different scales (or resolution) using a
% Laplacian operator given below:
%  [-1/8 -1/8 -1/8; -1/8 1 -1/8; -1/8 -1/8 -1/8];
% Apply the Laplacian operator to every level of the Gaussian pyramid generated in the previous assignment. (Use the images generated using Approach B in last assignment).
% Step 2: Segment the second order derivative image by assigning value 1 to all pixels of magnitude greater than 0 and value 0 to all pixels of magnitude less than or equal to zero.
% Step 3: Detect the zero crossing in the segmented image. This is done by tagging any pixel which has at least one neighbor who is of different value than the pixel itself.
% Step 4: Examine the pixels surrounding the zero crossing pixels in the second order derivative image. Calculate the local variance and mark it as an edge pixel if this value is greater than a certain threshold. This completes the edge detection.

LapOp=[-1/8 -1/8 -1/8; -1/8 1 -1/8; -1/8 -1/8 -1/8];

%Perfrom Convolution of gaussain levels with Laplacian Operation 
%start asgn 3 
    for i=0:7 % processing for layers

    %Step 1 : Convolution
    GPconv=convolve(eval(['G', num2str(i)]),LapOp,'same');
    GP_conv{j}{i+1}=GPconv;
    figure, imshow(GPconv);

    % Step 2 : segmentation
    [m n]=size(GPconv);
        for k=1:m
            for l=1:n
                if GPconv(k,l)>0
                  GP_Segmented{j}{i+1}(k,l)=1;
                else
                  GP_Segmented{j}{i+1}(k,l)=0;
                end

            end
        end
    %Step 3: Detection of zero crossing. for thsi search for change in
    %neighboring 8 pixels
    index=1;
    windowsize=3;
    threshold=30;
     
       for k=2:m-1 % Edge detection for a single layer 
         for l=2:n-1
                  if GP_Segmented{j}{i+1}(k-1,l-1)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold); 
                  elseif GP_Segmented{j}{i+1}(k-1,l)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold);
                  elseif GP_Segmented{j}{i+1}(k-1,l+1)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold);
                  elseif GP_Segmented{j}{i+1}(k,l+1)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold);
                  elseif GP_Segmented{j}{i+1}(k+1,l+1)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold);
                  elseif GP_Segmented{j}{i+1}(k+1,l)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold);
                  elseif GP_Segmented{j}{i+1}(k+1,l-1)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold);
                  elseif GP_Segmented{j}{i+1}(k,l-1)~=GP_Segmented{j}{i+1}(k,l)
                        GP_EdgeIndx{j}{i+1}(k,l)=255;
                        GP_EdgeIndx1{j}{i+1}(index)= {[k,l]};
                        index=index+1;
                        [GP_Edge{j}{i+1}(k,l), LocalVar]=funcVarianceMapMod(GPconv,k,l, windowsize, threshold);
                  end % end if

         end   % end of for with l
       end  % end of for with m 
     % End Edge detection for a single layer 
     
     %Write the Edge of the Image
     EdgeImg=GP_Edge{j}{i+1};
     figure, imshow(uint8(EdgeImg));
     cmap = colormap('gray');
     imwrite(EdgeImg,cmap, ['EdgeImg_Layer',num2str(i),ImageNames{j}], 'jpeg');
     pause(10);
     close all;
    end  % Processing for all layers
    
    
pause(10)
close all;

end % Processsing for all images 


