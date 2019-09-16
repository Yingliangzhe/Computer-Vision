function [EdgeDecision, var]=funcVarianceMapMod(inputImage, row, col, windowSize, thresh)
% Find out how many rows and columns are to the left/right/up/down of the
% central pixel based on the window size
win=(windowSize-1)/2;

% Compute a map of mean values

        sum=0.0;
        for(a=-win:1:win)
            for(b=-win:1:win)
                sum=sum+inputImage(row+a,col+b);
            end
        end
        LocalMean=sum/(windowSize*windowSize);

% Compute a map of variance values

        sum=0.0;
        for(a=-win:1:win)
            for(b=-win:1:win)
                sum=sum+((inputImage(row+a,col+b)-LocalMean)^2);
            end
        end         
        var=sum/((windowSize*windowSize)-1);
        % Apply threshold to produce a binarized variance map
        if (var > thresh)
            EdgeDecision = 255;
        else
            EdgeDecision = 0;
        end

