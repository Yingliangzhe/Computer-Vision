function best_x = find_best_match(patch,strip)
  %% load packages
  pkg load image;
  
  min_diff = Inf;
  best_x = 0; 
  for x = 1:(size(strip)(2) - size(patch)(2) + 1)
    other_patch = strip(:,x:(x + size(patch)(2) - 1));
    diff = sumsq((patch - other_patch)(:));
    if diff < min_diff
      min_diff = diff;
      best_x = x;
    endif
  endfor
end
