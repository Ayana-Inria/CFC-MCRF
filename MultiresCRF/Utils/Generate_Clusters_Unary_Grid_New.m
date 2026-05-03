function [ C, Cluster_unary, Cluster_variance ] = Generate_Clusters_Unary_Grid_New( x_tot, act, cluster_num, rand_grid, pix_step)
%Generate_Clusters_Unary_Grid_New Generates clusters from a tensor (image)
%using a Kmeans algorithm
%   Detailed explanation goes here

%should increase up to at least 1% of total pixels
%check for grid subsampling
noise_range = pix_step-1;

%generate sampling grid (with noise)
limit_x = size(x_tot,2)-pix_step;    limit_y = size(x_tot,1)-pix_step;
x = 1:pix_step:limit_x;
y = 1:pix_step:limit_y;
[X,Y] = meshgrid(x,y);
noise_x = randi(noise_range,size(X));
x_noisy = double(X) + double(noise_x);
noise_y = randi(noise_range,size(Y));
y_noisy = double(Y) + double(noise_y);

x_tot_r = reshape(x_tot, size(x_tot,1)*size(x_tot,2),size(x_tot,3));
linearInd = sub2ind(size(x_tot), y_noisy, x_noisy);

linearInd_r = reshape(linearInd, size(linearInd,1)*size(linearInd,2),1);
select_tot = x_tot_r(linearInd_r,:);

%spatial_w = 0.07;
border = 8;
[~,label] = max(act,[],3);
CLASS = reshape(label,1,size(label,1)*size(label,2));
actRow = reshape(act,size(act,1)*size(act,2),size(act,3))'; 
x_row = double(reshape(x_tot,size(x_tot,1)*size(x_tot,2),size(x_tot,3)));

fprintf('running kmeans...\n');
%[~,C] = kmeans(select_tot,cluster_num);
options_km = statset('UseParallel',1);
[~,C] = kmeans(select_tot,cluster_num,'MaxIter',500,'Options',options_km,'Replicates',8);
%'MaxIter',1

fprintf('assigning unarys...\n');
yf = size(C,1);
class_num = size(act,3)
Cluster_unary = (1/class_num)*ones(class_num,yf);
Cluster_variance = (1/class_num)*ones(class_num,yf);
[~,idx] = pdist2(C,x_row, 'euclidean','Smallest',1);
for i=1:size(C,1) %for each cluster
    belong_to_clust = find(idx==i);
    Cluster_unary(:,i) = (sum(actRow(:,belong_to_clust),2))/nnz(belong_to_clust);
    for c=1:class_num %6
        Cluster_variance(c,i) = var(actRow(c,belong_to_clust));
    end
end
%Cluster_unary = -log(Cluster_unary + eps);

end