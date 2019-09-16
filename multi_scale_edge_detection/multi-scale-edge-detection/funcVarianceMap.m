function [varianceImg, timeTaken]=funcVarianceMap(inputImage, windowSize, thresh)
try 
    % Grab the image information (metadata) of input image using the function imfinfo
    inputImageInfo=imfinfo(inputImage);
    if(getfield(inputImageInfo,'ColorType')=='truecolor')
    % Read an image using imread function, convert from RGB color space to
    % grayscale using rgb2gray function and assign it to variable inputImage
        inputImage=rgb2gray(imread(inputImage));
        % Convert the image from uint8 to double
        inputImage=double(inputImage);
    else if(getfield(inputImageInfo,'ColorType')=='grayscale')
    % If the image is already in grayscale, then just read it.        
            inputImage=imread(inputImage);
            % Convert the image from uint8 to double
            inputImage=double(inputImage);
        else
            error('The Color Type of Left Image is not acceptable. Acceptable color types are truecolor or grayscale.');
        end
    end
catch
    % if it is not an image but a variable
    inputImage=inputImage;
end
% Find the size (columns and rows) of the input image and assign the rows to
% variable nr, and columns to variable nc
[nr,nc] = size(inputImage);
% Check the size of window to see if it is an odd number.
if (mod(windowSize,2)==0)
    error('The window size must be an odd number.');
end
% Create an image of size nr and nc, fill it with zeros and assign
% it to variable meanImg
meanImg=zeros(nr, nc);
% Create an image of size nr and nc, fill it with zeros and assign
% it to variable varianceImg
varianceImg=zeros(nr, nc);
% Find out how many rows and columns are to the left/right/up/down of the
% central pixel based on the window size
win=(windowSize-1)/2;
tic; % Initialize the timer to calculate the time consumed.
% Compute a map of mean values
for(i=1+win:1:nr-win)
    for(j=1+win:1:nc-win)
        sum=0.0;
        for(a=-win:1:win)
            for(b=-win:1:win)
                sum=sum+inputImage(i+a,j+b);
            end
        end
        meanImg(i,j)=sum/(windowSize*windowSize);
    end
end
% Compute a map of variance values
for(i=1+win:1:nr-win)
    for(j=1+win:1:nc-win)
        sum=0.0;
        for(a=-win:1:win)
            for(b=-win:1:win)
                sum=sum+((inputImage(i+a,j+b)-meanImg(i,j))^2);
            end
        end         
        var=sum/((windowSize*windowSize)-1);
        % Apply threshold to produce a binarized variance map
        if (var > thresh)
            varianceImg(i,j) = 255;
        else
            varianceImg(i,j) = 0;
        end
    end
end
% Stop the timer to calculate the time consumed.
timeTaken=toc;