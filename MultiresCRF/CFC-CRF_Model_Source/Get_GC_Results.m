function [Image_Patch, Cluster_Data, new_img_global] = Get_GC_Results(conn, LABELS_global, Cluster_Data, Image_Patch, Features_Patch, Patch_division, Param)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%This fucntion takes the labeled matrix given by graph cut and extracts the
%patche removing the overlays (clusters, etc.). Any possible additional
%result should be taken from here.


    % plot(max(Clust_pairw_left,[],2));
    % plot(max(Clust_pairw_left_global,[],2));
    % plot(max(clust_pairw_global,[],2));
    % plot(max(clust_pairw,[],2));
    % plot(min(clust_pairw,[],2));
    % plot(min(clust_pairw_global,[],2));
    % imagesc(clust_pairw_global);
    % imagesc(clust_pairw);

    %[~,clus_pix_idx_global] = pdist2(C_global, x_row, 'euclidean','Smallest',1);
    % figure; imagesc(reshape(clus_pix_idx_global(1,:),[Patch_division.xsy Patch_division.xsx]));
    % colormap(rand(1000,3));

    connected_num = size(conn,1);
    Cluster_Data.GraphCut_Labels = LABELS_global(end-connected_num+1:end,:) + 1;

    [Cluster_Data.GraphCut_Pred] = Clusters_Labels_Indexes( Image_Patch.act, ...
        Cluster_Data.Centroids, Cluster_Data.GraphCut_Labels, connected_num, conn ); %clus_graph_global

%     clus_pix_class_graph = zeros(1,Patch_division.xsx*Patch_division.xsy);
%     for cn=1:connected_num
%         posit_global = (clus_pix_idx_global(1,:)==conn(cn));
%         clus_pix_class_graph(posit_global) = cluster_label_graph(cn);
%     end
%     clear cn posit_global
%     %clus_graph_global = zeros(Patch_division.xsy,Patch_division.xsx,3);
%     clus_graph_global = Assign_Color_to_Class_v2(reshape(clus_pix_class_graph,...
%                             Patch_division.xsy, Patch_division.xsx));% , clus_graph_global );
%     %figure; imshow(clus_graph_global);
%     clear clus_pix_class_graph

    [~,Cluster_Data.Initial_CNN_labels] = max(Cluster_Data.Posterior_Prob,[],1);
    all_clusters = 1:Param.Clust.f.cluster_num_global;
    [Cluster_Data.Initial_CNN_Pred] = Clusters_Labels_Indexes( Image_Patch.act, ...
        Cluster_Data.Centroids, Cluster_Data.Initial_CNN_labels, Param.Clust.f.cluster_num_global, all_clusters');

    if false
        figure; 
        h    = [];
        h(1)=subplot(2,3,1);
        imagesc(reshape(clus_pix_idx_global(1,:),[Patch_division.xsy Patch_division.xsx]), 'parent', h(1));
        colormap(rand(1000,3));
        h(2)=subplot(2,3,2);    imagesc(Cluster_Data.Initial_CNN_Pred, 'parent', h(2));
        h(3)=subplot(2,3,3);    imagesc(Cluster_Data.GraphCut_Pred, 'parent', h(3));
        h(4)=subplot(2,3,4);    imagesc(Image_Patch.Gt, 'parent', h(4));
        h(5)=subplot(2,3,5);    imagesc(((8*Image_Patch.act(:,:,1:3)/255+0.018*Cluster_Data.Initial_CNN_Pred)/8), 'parent', h(5));
        h(6)=subplot(2,3,6);    imagesc(((8*Image_Patch.act(:,:,1:3)/255+0.018*Cluster_Data.GraphCut_Pred)/8), 'parent', h(6));
        clear h;
    end

%     if true
%         figure; 
%         h    = [];
%         h(1)=subplot(1,3,1);
%         imagesc(reshape(clus_pix_idx_global(1,:),[Patch_division.xsy Patch_division.xsx]), 'parent', h(1));
%         colormap(rand(1000,3));
%         h(2)=subplot(1,3,2);    imagesc(Cluster_Data.Initial_CNN_Pred, 'parent', h(2));
%         h(3)=subplot(1,3,3);    imagesc(Cluster_Data.GraphCut_Pred, 'parent', h(3));
%         clear h;
%     end
    
    %clear clus_pix_idx_global conn connected_num
    %to do next: passing just the cluster inside a graph cut and check how the
    %result looks like. try also to consider all the clusters and not only the
    %connected ones (maybe this can reduce variance of the algo).

    %figure; imagesc(reshape(Multiscale_Tensor,size(Multiscale_Tensor,1)*size(Multiscale_Tensor,2),size(Multiscale_Tensor,3)));

    % lab_vect_global{2} = LABELS_global(size(actRow,2)+1:size(actRow,2)+size(actRow,2)/4,:) + 1;
    % lab_vect_global{3} = LABELS_global((size(actRow,2)+size(actRow,2)/4)+1:(size(actRow,2)+...
    %                                     size(actRow,2)/4)+size(actRow,2)/16,:) + 1;
    % 
    % new_img_global{2} = reshape(lab_vect_global{2}, [size(x,1)/2 size(x,2)/2]);
    % lay_pred = Assign_Color_to_Class( new_img_global{2}, new_img_global{2} );
    % figure; imagesc(lay_pred);
    % 
    % new_img_global{3} = reshape(lab_vect_global{3}, [size(x,1)/4 size(x,2)/4]);
    % lay3_pred = Assign_Color_to_Class( new_img_global{3}, new_img_global{3} );
    % figure; imagesc(lay3_pred);

    Image_Patch.Gt = Image_Patch.Gt(1:Patch_division.xsy,1:Patch_division.xsx,:);
    lab_vect_global = cell(Param.Load.scale_num,1);
    %lab_vect_global{1} = LABELS_global(1:size(UNARY{1},2),:) + 1;
    lab_vect_global{1} = LABELS_global(1:(size(Features_Patch.prediction,1)*size(Features_Patch.prediction,2)),:) + 1;
    new_img_global = cell(Param.Load.scale_num,1);
    %new_img_global{1} = reshape(lab_vect_global{1}, [size_x_1 size_x_2]);
    new_img_global{1} = reshape(lab_vect_global{1}, [size(Image_Patch.act,1) size(Image_Patch.act,2)]);
    %clear size_x_1 size_x_2
    clear lab_vect_global
    Image_Patch.CNN_init_pred = Assign_Color_to_Class_PRISMA_ER(Image_Patch.CNN_Init_Labels); %initial prediction
    Image_Patch.Graph_pred = Assign_Color_to_Class_PRISMA_ER(new_img_global{1});
    %[~, ~] = Plot_Compared_Results( Image_Patch.CNN_init_pred, Image_Patch.Graph_pred, Image_Patch.Gt );
    
    %[init_accuracy_global, accuracy_global] = Plot_Compared_Results( init_pred, graph_pred_global, gt );
    clear init_accuracy_global accuracy_global

end

