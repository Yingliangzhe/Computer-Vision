function [OutImg] = convolve(I, F, str)
%function [OutImg] = convolve(I, F, iw, ih, fw, fh)

% This function performs convolution of gray scale image I with the filter
% F. 
% I is a gray scale image. This is a two dimensional array of size iw by ih. 
% F denotes a filter. This is a two dimensional array of floating point 
% numbers of size fw by fh. Usually ( fh, fw ) << ( ih, iw). The output 
% O(x, y) is an image of same size as I. The value of O at any pixel is 
% generated % by position F on I(x, y) such that the top right pixel of F 
% coincides with I(x, y), and then multiplying the values of I and F 
% for all the pixels of I covered by F, and finally summing these values.


%compute the dimension of the images
[ih, iw]=size(I);
[fh, fw]=size(F);
%The dimension of the convolution output O
%Ox=max(Ix+Fx-1, Ix, Fx);
%Oy=max(Iy+Fy-1, Iy, Fy);

OutImg=zeros(ih, iw);

for iy=1:ih  % image hight or rows
    for ix=1:iw  % image width or col
        pixelSum=0;
        
        fy = 1;
		fx = fw;
        
        for y=1:fh  % kernel hight or rows
            for x=1:fw % kernel width or columns
                        if(iy + y -1<= ih  && ix - x +1>=1) 
							pixelSum =pixelSum + I(iy+y-1, ix-x+1)*F(fy+y-1,fx-x+1);
                        end         
                               
            end
        end
        OutImg(iy,ix)=pixelSum;
    end
end 

end


