function [ patches_indy, patches_indx ] = Patches_Starting_Points_Pots_v3(xsyf, xsxf, crop_size, border )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

%xsyf = size(features_full.prediction,1); xsxf = size(features_full.prediction,2);

% xsyfc = size(features_full.feature_concat2,1);
% xsxfc = size(features_full.feature_concat2,2);
% if xsyf~=xsyfc || xsxf~=xsxfc
%     msg = 'Dimension of prediction is not consistent with ones of intermediate layers';
%     error(msg)
% end
    
xsy = crop_size; xsx = crop_size;

num_patch_y = floor((xsyf-2*border)/(crop_size-2*border));
num_patch_x = floor((xsxf-2*border)/(crop_size-2*border));

rem_y = (rem((xsyf-2*border),(crop_size-2*border))~=0);
rem_x = (rem((xsxf-2*border),(crop_size-2*border))~=0);

patches_indx = zeros(num_patch_x+double(rem_x),1); %patches_indx = zeros(num_patch_x+1,1);%zeros((num_patch_y+1),(num_patch_x+1));
patches_indy = zeros(num_patch_y+double(rem_y),1); %patches_indy = zeros(num_patch_y+1,1);
for r=1:num_patch_y
    offset_y = ((r-1)*(xsy-(border*2)));%-(ceil((r-1)/r)*border);
    patches_indy(r,1) = offset_y;
end
if rem_y
    patches_indy(end,1) = xsyf-crop_size;
end

for c=1:num_patch_x
    offset_x = ((c-1)*(xsx-(border*2)));%-(ceil((c-1)/c)*border);
    patches_indx(c,1) = offset_x;
end
if rem_x
    patches_indx(end,1) = xsxf-crop_size;
end

end

