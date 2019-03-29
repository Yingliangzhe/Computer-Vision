function result = select_gdir(gmag,gdir,mag_min,angle_low,angle_high)
  result = gdir>=angle_low & gdir<=angle_high & gmag >= mag_min;
end
