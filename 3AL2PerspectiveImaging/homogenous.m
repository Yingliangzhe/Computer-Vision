
function p_img = homogenous(p,f)
  p = [p 1];
  trans_matrix = [1 0 0 0;0 1 0 0;0 0 1/f 0];
  p_img_temp = trans_matrix*p';
  p_img = [p_img_temp(1)/p_img_temp(3) p_img_temp(2)/p_img_temp(3)]';
end

