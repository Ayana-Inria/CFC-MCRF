function [ UNARY_tot, CLASS_tot, PAIRWISE_tot ] = Generate_Energy_MR(patch, UNARY, CLASS, PAIRWISE, lambda)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


UNARY_tot = cat(2, UNARY.f, UNARY.c);
CLASS_tot = cat(2, CLASS.f, CLASS.c);
total_size = size(UNARY_tot, 2);

Pairwise_MR = lambda*Generate_MR_pairwise( patch.f, patch.c );  
PAIRWISE_tot = sparse(total_size, total_size);
PAIRWISE_tot(1:size(PAIRWISE.f, 1), 1:size(PAIRWISE.f, 1)) = PAIRWISE.f;
PAIRWISE_tot(size(PAIRWISE.f, 1)+1:end, size(PAIRWISE.f, 1)+1:end) = PAIRWISE.c;
PAIRWISE_tot(1:size(Pairwise_MR, 1), size(PAIRWISE.f, 1)+1:size(PAIRWISE.f, 1)+size(Pairwise_MR, 2)) = Pairwise_MR;
PAIRWISE_tot(size(PAIRWISE.f, 1)+1:size(PAIRWISE.f, 1)+size(Pairwise_MR, 2), 1:size(Pairwise_MR, 1)) = Pairwise_MR';