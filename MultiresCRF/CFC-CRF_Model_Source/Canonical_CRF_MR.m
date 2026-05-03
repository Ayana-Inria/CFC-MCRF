function [CLASS_tot_global, UNARY_tot_global, Pairw_tot_global, LABELCOST, sigma, Image_Patch, diff_tot] = ...
                Canonical_CRF_MR(Param, Image_Patch, Features_Patch, lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%subcalls:
%Neighbours_diff_v2
%Generate_Image_Pairwise

    sigma = struct('pp', 800, 'cc', 800, 'pc', 800);

    diff = struct('above', cell(Param.Load.scale_num,1), 'below', cell(Param.Load.scale_num,1),...
                   'left', cell(Param.Load.scale_num,1), 'right', cell(Param.Load.scale_num,1));
    
    [diff(1).above, diff(1).below, diff(1).left, diff(1).right] =...
                            Neighbours_diff_v2(Image_Patch.act, sigma(1).pp, false);

    %diff_tot is used only for visualization
    diff_tot = cell(3,1);
    diff_tot{1} = diff(1).above(1:end-1,2:end-1) + diff(1).below(2:end,2:end-1) ...
                + diff(1).left(2:end-1,1:end-1) + diff(1).right(2:end-1,2:end);

    %tic;
    fprintf('scale 1...\n');
    PAIRWISE = cell(Param.Load.scale_num,1);
    CLASS = cell(Param.Load.scale_num,1);
    UNARY = cell(Param.Load.scale_num,1);

    act = Features_Patch.prediction;
    [~,Image_Patch.CNN_Init_Labels] = max(act,[],3); %[valMax,label] = max(act,[],3);
    CLASS{1} = reshape(Image_Patch.CNN_Init_Labels,1,size(Image_Patch.CNN_Init_Labels,1)*size(Image_Patch.CNN_Init_Labels,2));

    actRow = reshape(act,size(act,1)*size(act,2),size(act,3))';
    class_num = size(act, 3);
    clear act
    UNARY{1} = -log(actRow + eps);
    clear actRow
    LABELCOST = single(ones(class_num) - eye(class_num));

    size_x_1 = size(Image_Patch.act,1);   size_x_2 = size(Image_Patch.act,2);
    fprintf('generating pairwise...\n');
    PAIRWISE{1} = Generate_Image_Pairwise(Image_Patch.act, diff(1)); %pixels pairwise of img
    Pairw_tot_global = lambda.pp.*PAIRWISE{1};
    clear index

    CLASS_tot_global = CLASS{1};
    UNARY_tot_global = UNARY{1};
    clear ind_matr
    %clear diff_tot
end

