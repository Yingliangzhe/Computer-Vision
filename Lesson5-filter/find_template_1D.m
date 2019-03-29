

function index = find_template_1D(t,s)
  % load a package
  pkg load image;
  
  %% using the normalized cross correlation function 
  c = normxcorr2(t,s);
  %% max function can also return the index of a array element
  [maxValue rawIndex] = max(c);
  index = rawIndex - size(t,2) + 1;
  disp(index);
end

