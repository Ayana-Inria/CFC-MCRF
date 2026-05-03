folder = fileparts(which('multires_CFCCRF')); 
addpath(genpath(strcat(folder,'\CFC-CRF_Model_Source')));
addpath(genpath(strcat(folder,'\Utils')));
addpath(genpath(strcat(folder,'\gcmex-2.3.0')));

elapsed_time = zeros(144,1);
%Set the criteria for diving the image in patches
% resolution ratio between the coarse and the fine resolution images
res_ratio = double(size(imageTensor.feature_f,1) / size(imageTensor.feature_c,1));
img_f_size_y = size(imageTensor.feature_f,1); img_f_size_x = size(imageTensor.feature_f,2);
img_c_size_y = img_f_size_y / res_ratio; img_c_size_x = img_f_size_x / res_ratio;
[Patch_division.f.indy, Patch_division.f.indx ] = Patches_Starting_Points_Pots_v3...
      (img_f_size_y, img_f_size_x, Patch_division.f.xsx, Patch_division.f.border );
Patch_division.c.border = double(Patch_division.f.border / res_ratio);
Patch_division.c.xsx = double(Patch_division.f.xsx / res_ratio);
Patch_division.c.xsy = double(Patch_division.f.xsy / res_ratio);

[Patch_division.c.indy, Patch_division.c.indx ] = Patches_Starting_Points_Pots_v3...
      (img_c_size_y, img_c_size_x, Patch_division.c.xsx, Patch_division.c.border);

Image_Patch = struct();
Image_Patch.f = struct('offset_x',[],'offset_y',[], 'act',[],'Gt',[]);
Image_Patch.c = struct('offset_x',[],'offset_y',[], 'act',[],'Gt',[]);

if Param.Save.save_data
    Save_folder = 'res';
    if ~isfolder(Save_folder)
        mkdir(Save_folder);
    end
    % Get the list of subfolders in "res"
    subfolders = dir(Save_folder);
    % Filter out "." and ".." directories
    subfolders = subfolders([subfolders.isdir]);
    % Extract the numbers from the folder names
    folder_numbers = [];
    for i = 1:length(subfolders)
        folder_name = subfolders(i).name;
        if ~isempty(regexp(folder_name, '^\d+$', 'once')) % Check if the folder name is a number
            folder_numbers = [folder_numbers, str2double(folder_name)];
        end
    end
    % Determine the next folder number
    if isempty(folder_numbers)
        next_folder_num = 0; % If no subfolders exist, start with 0
    else
        next_folder_num = max(folder_numbers) + 1; % Otherwise, increment the highest folder number by 1
    end
    % Define the new subfolder name
    Save_folder = fullfile(Save_folder, num2str(next_folder_num));
    % Create the new subfolder
    mkdir(Save_folder);
end

%% Run Cl-FC-CRF on each patch separately (the process can be parallelized)
for r=1:size(Patch_division.f.indy,1)
Image_Patch.f.offset_y = Patch_division.f.indy(r,1);
Image_Patch.c.offset_y = Patch_division.c.indy(r,1);

for c=1:size(Patch_division.f.indx,1)   %c=1:7:8
    Image_Patch.f.offset_x = Patch_division.f.indx(c,1);
    Image_Patch.c.offset_x = Patch_division.c.indx(c,1);

    fprintf('patch_fine_row_%i_col_%i...\n',Image_Patch.f.offset_y,Image_Patch.f.offset_x);
    fprintf('patch_coarse_row_%i_col_%i...\n',Image_Patch.c.offset_y,Image_Patch.c.offset_x);

    % Select image patch of the fine resolution image
    [Features_Patch.f, Image_Patch.f] = Select_crop_v4_Indexing_MR(posteriors.f,...
       imageTensor.feature_f, imageTensor.gt_f, Patch_division.f, Image_Patch.f);
    Image_Patch.f.act = cast(Image_Patch.f.act, 'double');

    % Select image patch of the coarse resolution image
    [Features_Patch.c, Image_Patch.c] = Select_crop_v4_Indexing_MR(posteriors.c,...
       imageTensor.feature_c, imageTensor.gt_c, Patch_division.c, Image_Patch.c);
    Image_Patch.c.act = cast(Image_Patch.c.act, 'double');

    lambda = struct();
    UNARY_tot_global = struct();
    Pairw_tot_global = struct();
    CLASS_tot_global = struct();
    lambda.f = struct('pp', 2, 'cc', 1, 'pc', 1);
    lambda.c = struct('pp', 2, 'cc', 1, 'pc', 1);
    lambda.mr = 1;
    
    fprintf('lambda fine pixel-pixel = %d; lambda cluster-cluster = %d; lambda pixel-cluster = %d\n', lambda.f.pp, lambda.f.cc, lambda.f.pc);
    fprintf('lambda coarse pixel-pixel = %d; lambda cluster-cluster = %d; lambda pixel-cluster = %d\n', lambda.c.pp, lambda.c.cc, lambda.c.pc);

    % Define the graph for the cannonical CRF for the fine resolution image
    [CLASS_tot_global.f, UNARY_tot_global.f, Pairw_tot_global.f, LABELCOST, sigma.f, Image_Patch.f, diff_tot.f] = ...
            Canonical_CRF_MR(Param, Image_Patch.f, Features_Patch.f, lambda.f);
    % Define the graph for the cannonical CRF for the coarse resolution image
    [CLASS_tot_global.c, UNARY_tot_global.c, Pairw_tot_global.c, LABELCOST, sigma.c, Image_Patch.c, diff_tot.c] = ...
            Canonical_CRF_MR(Param, Image_Patch.c, Features_Patch.c, lambda.c);

    % Add cluster level connection at fine resolution
    fprintf('cluster model...\n');
    [Pairw_tot_global.f, CLASS_tot_global.f, UNARY_tot_global.f, conn.f, Cluster_Data.f] = ...
        Cluster_Connected_Model_v2(Pairw_tot_global.f, CLASS_tot_global.f, UNARY_tot_global.f, ...
                                Cluster_Data.f, Image_Patch.f.act, sigma.f, lambda.f,...
                                Param.Clust.f);
    UNARY_tot_global.f = cast(UNARY_tot_global.f,'single');

    % Add cluster level connection at coarse resolution
    fprintf('cluster model...\n');
    [Pairw_tot_global.c, CLASS_tot_global.c, UNARY_tot_global.c, conn.c, Cluster_Data.c] = ...
        Cluster_Connected_Model_v2(Pairw_tot_global.c, CLASS_tot_global.c, UNARY_tot_global.c, ...
                                Cluster_Data.c, Image_Patch.c.act, sigma.c, lambda.c,...
                                Param.Clust.c);
    UNARY_tot_global.c = cast(UNARY_tot_global.c,'single');
    tic;


    % Create the final multiresolution pairwise and unary terms
    [UNARY_tot_global, CLASS_tot_global, Pairw_tot_global] = Generate_Energy_MR(Patch_division, UNARY_tot_global, CLASS_tot_global, Pairw_tot_global, lambda.mr);

    fprintf('running graph cut...\n');
    curDir = pwd;
    [LABELS_global,~,~] = GCMex(CLASS_tot_global-1, UNARY_tot_global, Pairw_tot_global, LABELCOST);
    cd(curDir);
    clear curDir
    t_gc_global=toc;
    elapsed_time(12*(r-1)+c,1) = toc;
    fprintf('graph cut with global clusters done...y=%i , x=%i...\n',Image_Patch.f.offset_y,Image_Patch.f.offset_x);
    
    [Image_Patch.f, Cluster_Data.f, new_img_global.f] = Get_GC_Results(conn.f, LABELS_global(1:360064,1), ...
        Cluster_Data.f, Image_Patch.f, Features_Patch.f, Patch_division.f, Param);
    
    if Param.Save.save_data
        name = strcat('img_f_',sprintf('y%i_x%i',Image_Patch.f.offset_y,Image_Patch.f.offset_x));
        curDir = pwd;
        cd(Save_folder);
        write_file(Param, Save_folder, sigma, lambda)
        save(name,'new_img_global');
        cd(curDir)
        clear name;
        clear curDir;
    end
    time = toc;
end
end

curDir = pwd;
cd(Save_folder);
elaps_time_name = 'elapsed_time';
save(elaps_time_name,'elapsed_time');



