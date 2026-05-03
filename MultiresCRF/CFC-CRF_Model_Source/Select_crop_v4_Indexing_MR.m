function [Features_Patch, Image_Patch] = Select_crop_v4_Indexing_MR(...
            posteriors, imageTensor, gt, Patch_division, Image_Patch)
            %features_full, x, gt, dsm, x_tot_global, xsy, xsx, offset_y, offset_x, scale_num, Pooling_index )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%Cuts te atch to process, based on Patch_division

xsy = Patch_division.xsy;
xsx = Patch_division.xsx;
offset_y = Image_Patch.offset_y;
offset_x = Image_Patch.offset_x;

Features_Patch.prediction = posteriors(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);

                                           
Image_Patch.act = imageTensor(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);
Image_Patch.Gt  = gt(offset_y+1:xsy+offset_y,offset_x+1:xsx+offset_x,:);

end

