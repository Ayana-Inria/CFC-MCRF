function [ PAIRWISE ] = Generate_Image_Pairwise( x, diff )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%vector index whose length is the one of the image                   
index = 1:1:(size(x,1)*size(x,2));
%reshape it into a matrix
ind_matr = reshape(index, [size(x,1) size(x,2)]);
%those are the matrices with indexes of the pixels of which we want to know
%the difference with respect to neighbours. so those which go in y
%coordinate of transition matrix. first element of 
i_a = ind_matr(2:size(ind_matr,1),1:size(ind_matr,2),:); %without first row
i_b = ind_matr(1:size(ind_matr,1)-1,1:size(ind_matr,2),:); %without last r
i_r = ind_matr(1:size(ind_matr,1),1:size(ind_matr,2)-1,:);
i_l = ind_matr(1:size(ind_matr,1),2:size(ind_matr,2),:); %without first col

%those are indexes of pixels with which we computed the differences. 
ind_matr_a = ind_matr(1:size(ind_matr,1)-1,1:size(ind_matr,2));
ind_matr_b = ind_matr(2:size(ind_matr,1),  1:size(ind_matr,2));
ind_matr_r = ind_matr(1:size(ind_matr,1),  2:size(ind_matr,2));
ind_matr_l = ind_matr(1:size(ind_matr,1),  1:size(ind_matr,2)-1);


%sparse(index_i, inde_j, element_to_assign_in_ij, size, size
sC2 = size(x,1)*size(x,2);%size(CLASS,2);
PAIRWISE = sparse(ind_matr_a(:),i_a(:),diff.above(:),sC2,sC2);
PAIRWISE = PAIRWISE + sparse(ind_matr_b(:),i_b(:),diff.below(:),sC2,sC2);
PAIRWISE = PAIRWISE + sparse(ind_matr_r(:),i_r(:),diff.right(:),sC2,sC2);
PAIRWISE = PAIRWISE + sparse(ind_matr_l(:),i_l(:),diff.left(:), sC2,sC2);
%it is possible to invert diff.above with diff.below (and left with right)
%since they are exactly the same.

%up to now the transition matrix of x is computed
end

