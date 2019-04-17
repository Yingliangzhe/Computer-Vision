% TODO: Match two strips to compute disparity values
function disparity = match_strips(strip_left, strip_right, b)
    % For each non-overlapping patch/block of width b in the left strip,
    %   find the best matching position (along X-axis) in the right strip.
    % Return a vector of disparities (left X-position - right X-position).
    % Note: Only consider whole blocks that fit within image bounds.
    pkg load image;
    
    num_blocks = floor(size(strip_left,2)/b);
    disparity = zeros([1 num_blocks]);
    for block = 0 : (num_blocks - 1)
      x_left = block*b + 1;
      patch_left = strip_left(:,x_left:(x_left+b-1));
      x_right = find_best_match(patch_left,strip_right);
      disparity(1,block + 1) = (x_left - x_right);
    endfor
endfunction