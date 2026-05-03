function [Pairwise_tot, CLASS_tot, UNARY_tot, conn, Cluster_Data] = ...
        Cluster_Connected_Model_v2(Pairw_tot, CLASS_tot, UNARY_tot, ...
                    Cluster_Data, Tensor_Patch, sigma, lambda, Clust_Param)
%CLUSTER_CONNECTED_MODEL Creates unary for clusters and pairwise through them and between clusters and pixels.
%   This function takes in input Unary and Pairwise matrices of whatever model (pyramidal or not) and adds clusters on top of it.
%	

Tensor_Patch_Row = double(reshape(Tensor_Patch,...
    size(Tensor_Patch,1)*size(Tensor_Patch,2),size(Tensor_Patch,3)));

Cluster_Data.Unary = double(Clust_Param.unary_multiplier)*...
    (-log(Cluster_Data.Posterior_Prob + eps)); %eps=2.2204e-16 to avoid log(0)

%Pdist2 returns the distance between each pair of observations in X and Y using the metric specified by Distance.
%[D,I] = pdist2(___,Name,Value) also returns the matrix I using any of the arguments in the previous syntaxes. 
% The matrix I contains the indices of the observations in X corresponding to the distances in D.
[clus_pix_dist,clus_pix_idx] = pdist2(Cluster_Data.Centroids, Tensor_Patch_Row,...
                        'euclidean','Smallest',Clust_Param.cluster_neighb);

ind2 = 1:1:size(clus_pix_idx,2);
Clust_pairw_left = sparse(size(Cluster_Data.Centroids,1),size(clus_pix_idx,2));
for n=1:Clust_Param.cluster_neighb
    Clust_pairw_left = Clust_pairw_left + ...
    sparse(clus_pix_idx(n,:), ind2, exp(-(clus_pix_dist(n,:).^2)/sigma(1).pc),...
           size(Cluster_Data.Centroids,1),size(clus_pix_idx,2));
end

conn = find(max(Clust_pairw_left,[],2));
Clust_pairw_left = lambda.pc*Clust_pairw_left(conn,:);
Clust_pairw_up = Clust_pairw_left';

Clusters_Centroids_conn = Cluster_Data.Centroids(conn,:);
diff_cc_conn = pdist2(Clusters_Centroids_conn,Clusters_Centroids_conn);
clust_pairw_conn = exp(-(diff_cc_conn.^2)/sigma(1).cc);
clust_pairw_conn = lambda.cc*(clust_pairw_conn - eye(size(Clusters_Centroids_conn,1)));

Pairwise_tot = Pairw_tot;
%match to size of the model, in case there is already a pyramid
filler = sparse((size(Pairwise_tot,1)-size(Clust_pairw_up,1)),(size(Clust_pairw_up,2))); 
Pairwise_tot = cat(2, Pairwise_tot, cat(1, Clust_pairw_up, filler));
Pairw_temp = cat(2, Clust_pairw_left, sparse(size(Clust_pairw_up,2),...
                    (size(Pairwise_tot,1)-size(Clust_pairw_up,1))));
Pairwise_tot = cat(1, Pairwise_tot, cat(2, Pairw_temp, clust_pairw_conn));

Cluster_unary_conn = Cluster_Data.Unary(:,conn);
[~,Cluster_class] = max(Cluster_Data.Posterior_Prob(:,conn),[],1);

CLASS_tot = cat(2, CLASS_tot, Cluster_class);
UNARY_tot = cat(2, UNARY_tot, Cluster_unary_conn);
end

