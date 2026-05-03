function [clust_class_RGB] = Clusters_Labels_Indexes( x_tot_global, C_global,...
                                    cluster_label_global, cluster_num_global, connected_subset )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

x_row_global = double(reshape(x_tot_global,size(x_tot_global,1)*size(x_tot_global,2),size(x_tot_global,3)));
[~,clus_pix_idx_global] = pdist2(C_global, x_row_global, 'euclidean','Smallest',1);%cluster_neighb);

% figure; imagesc(reshape(clus_pix_idx_global(1,:),[size(x_tot_global,1) size(x_tot_global,2)]));
% colormap(rand(1000,3));

%[~,cluster_label_global] = max(Cluster_prob_global,[],1);
            
clus_pix_class_global = zeros(1,size(clus_pix_idx_global,2));

for c=1:cluster_num_global
    posit_global = (clus_pix_idx_global(1,:)==connected_subset(c));
    clus_pix_class_global(posit_global) = cluster_label_global(c);
end

clust_class = reshape(clus_pix_class_global,size(x_tot_global,1),size(x_tot_global,2));
clust_class_RGB = Assign_Color_to_Class_PRISMA_ER(clust_class);

%visual_clust = (8*x_tot_global(:,:,1:3)/255+0.018*clust_class)/8;
end

