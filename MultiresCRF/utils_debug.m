% Initialize random seed for reproducibility (optional)
rng('shuffle');



% Define the paths and parameters
main_folder = fullfile(pwd, '..'); 

dataset = 'SP';
CNN_model = 'FCN_SS';
save_dir = strcat(main_folder,'/res/', dataset, '/', CNN_model);
cluster_dir = strcat(main_folder,'/PRISMA_Clusters/', dataset, '/', CNN_model);
data_dir = strcat(main_folder,'/PRISMA_Tensors/', dataset, '/', CNN_model);
gt_dir = strcat(main_folder, '/PRISMA_Tensors/', dataset);

% Generate random images
% image_pan = rand(720, 720, 1);
% image_hys = rand(120, 120, 40);
image_pan = load(strcat(data_dir,'/',dataset,'_',CNN_model,'_act_8_PAN.mat')).act_pan;
image_hys = load(strcat(data_dir,'/',dataset,'_',CNN_model,'_act_8_HYS.mat')).act_hys;


% Generate random ground truth for gt_pan and resize to gt_hys dimensions
% gt_pan = randi([0, 4], 720, 720, 1, 'uint8'); %720
% gt_hys = imresize(gt_pan, [120, 120], 'nearest');
%gt_pan = imread(strcat(gt_dir,'/gt.tif'));
%gt_hys = imresize(gt_pan, [size(image_hys, 1), size(image_hys, 2)], 'nearest');
gt_pan = load(strcat(gt_dir,'/gt.mat')).gt;
gt_hys = load(strcat(gt_dir,'/gt_c.mat')).gt;

% Generate posteriors and normalize
% post_pan = rand(720, 720, 5);
% post_hys = rand(120, 120, 5);
% post_pan = post_pan ./ sum(post_pan, 3);
% post_hys = post_hys ./ sum(post_hys, 3);

post_pan = load(strcat(data_dir,'/',dataset,'_',CNN_model,'_post6_PAN.mat')).post_pan;
post_hys = load(strcat(data_dir,'/',dataset,'_',CNN_model,'_post6_HYS.mat')).post_hys;

% already softmaxxed
% post_pan = softmax(post_pan, 3);
% post_hys = softmax(post_hys, 3);

% Store data in structs
posteriors.f = post_pan;
posteriors.c = post_hys;
imageTensor.feature_f = image_pan;
imageTensor.feature_c = image_hys;
imageTensor.gt_f = gt_pan;
imageTensor.gt_c = gt_hys;


% Initialize Patch_division struct
Patch_division = struct();

% Define 'f' field with parameters
Patch_division.f = struct( ...
    'xsy', 600, ...
    'xsx', 600, ...
    'border', 60, ...
    'indy', 0, ... % Placeholder for the result of Patches_Starting_Points_Pots_v3
    'indx', 0 ...  % Placeholder for the result of Patches_Starting_Points_Pots_v3
);

% Define 'c' field with parameters
Patch_division.c = struct( ...
    'xsy', 0, ... % Placeholder
    'xsx', 0, ... % Placeholder
    'border', 0, ... % Placeholder
    'indy', 0, ... % Placeholder for the result of Patches_Starting_Points_Pots_v3
    'indx', 0 ...  % Placeholder for the result of Patches_Starting_Points_Pots_v3
);


Param = struct('Save',[],'Tensor',[],'Clust',[],'Load',[]);
Param.Save = struct('save_data', true, 'model', [], 'gtType', []);
if ~Param.Save.save_data
    sprintf('will not save results')
end
Param.Tensor = struct('useRGB', true, 'useDsm', true, 'normalize', true, ...
                      'spatial_weight', 0.05, 'preprocessing', false,  'band_weight', [0.9,1,1,0.8]);
                 
Param.Clust.f = struct('use_clusters', true, 'cluster_num_global', 64, 'rand_clustering', true,...
                     'cluster_neighb', 4, 'normalize', true, 'spatial_weight', 0.05,...
                     'unary_multiplier', 5625, 'c_scale_num', []);
Param.Clust.c = struct('use_clusters', true, 'cluster_num_global', 8, 'rand_clustering', true,...
                     'cluster_neighb', 4, 'normalize', true, 'spatial_weight', 0.05,...
                     'unary_multiplier', 1250, 'c_scale_num', []);
                 
Param.Load = struct('scale_num', 1, 'concat', true, 'feat_x', true, 'eroded', true);
Param.Clust.c_scale_num = Param.Load.scale_num;


addpath('./Utils/');
fprintf('computing clusters...\n');
[Cluster_Data.f.Centroids, Cluster_Data.f.Posterior_Prob, Cluster_Data.f.Variance] = ...
    Generate_Clusters_Unary_Grid_New(imageTensor.feature_f, posteriors.f,...
            Param.Clust.f.cluster_num_global, Param.Clust.f.rand_clustering, 100);

[Cluster_Data.c.Centroids, Cluster_Data.c.Posterior_Prob, Cluster_Data.c.Variance] = ...
    Generate_Clusters_Unary_Grid_New(imageTensor.feature_c, posteriors.c,...
            Param.Clust.c.cluster_num_global, Param.Clust.c.rand_clustering, 10);

% Save the clusters    
name_cl = strcat(cluster_dir, '/', sprintf('Clusters_img_%s_%s_%dclF_%dclC', dataset, CNN_model, Param.Clust.f.cluster_num_global, Param.Clust.c.cluster_num_global));

% Check if file exists and save accordingly
cd(cluster_dir);  % Change to the cluster directory

if exist(name_cl, 'file') == 2
    name_cl_new = sprintf('%s_new', name_cl);
    save(name_cl_new, 'Cluster_Data', '-v7.3');
else
    save(name_cl, 'Cluster_Data', '-v7.3');
end

cd(pwd);  % Return to the original directory