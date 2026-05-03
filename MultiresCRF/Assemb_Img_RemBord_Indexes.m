
folder = fileparts(which('Assemb_Img_RemBord_Indexes')); 
addpath(genpath(strcat(folder,'\CFC-CRF_Model_Source')));
addpath(genpath(strcat(folder,'\Utils')));
addpath(genpath(strcat(folder,'\gcmex-2.3.0')));

cur_dir = pwd;
cd(cur_dir);
img_size_y = size(imageTensor.feature_f,1);
img_size_x = size(imageTensor.feature_f,2);

cd(strcat(cur_dir,'/res/48'));

img = sparse(img_size_y,img_size_x);
Patch_division = struct('xsy',600,'xsx',600, 'border',60, 'indy',[],'indx',[]);
[Patch_indy, Patch_indx ] = Patches_Starting_Points_Pots_v3(img_size_y, img_size_x, Patch_division.xsx, Patch_division.border );

border = Patch_division.border;
complete_img = zeros(img_size_y,img_size_x,'uint8');
xsx = Patch_division.xsx;
xsy = Patch_division.xsy;

Cut_indy = struct('start',zeros(size(Patch_indy,1),1),'end',zeros(size(Patch_indy,1),1));
Write_indy = struct('start',zeros(size(Patch_indy,1),1),'end',zeros(size(Patch_indy,1),1));

Cut_indx = struct('start',zeros(size(Patch_indx,1),1),'end',zeros(size(Patch_indx,1),1));
Write_indx = struct('start',zeros(size(Patch_indx,1),1),'end',zeros(size(Patch_indx,1),1));

tic;
for r=1:size(Patch_indy,1)
    NotFirstR = ~(rem(r,size(Patch_indy,1))==1); %nella prima colonna la label y parte da 1 anziche da border
    lastR = (rem(r,size(Patch_indy,1))==0);
    inizioUltimaPatch_y = lastR*(Patch_division.xsy-border-(img_size_y-Patch_indy(end-1)+border-Patch_division.xsy));
    
    Cut_indy.start(r) = (border*NotFirstR)+inizioUltimaPatch_y+1;
    Cut_indy.end(r) = xsy-(border*(1-lastR));

    Write_indy.start(r) = Patch_indy(r)+(border*NotFirstR)+1+lastR*(Cut_indy.start(r)-border-1);
    Write_indy.end(r) = Write_indy.start(r)+(Cut_indy.end(r) - Cut_indy.start(r)); %end_w_y = start_w_y+(end_y - start_y)+1-1;
    
    for c=1:size(Patch_indx,1)
        NotFirstC = ~(rem(c,size(Patch_indx,1))==1); %cosi da 0 se e' la prima
        lastC = (rem(c,size(Patch_indx,1))==0); %da 1 se e' l'ultima
        inizioUltimaPatch_x = lastC*(Patch_division.xsx-border-(img_size_x-Patch_indx(end-1)+border-Patch_division.xsx));
        
        Cut_indx.start(c) = (border*NotFirstC)+inizioUltimaPatch_x+1;
        Cut_indx.end(c) = xsx-(border*(1-lastC));
        
        Write_indx.start(c) = Patch_indx(c)+(border*NotFirstC)+1+lastC*(Cut_indx.start(c)-border-1);
        Write_indx.end(c) = Write_indx.start(c)+(Cut_indx.end(c) - Cut_indx.start(c));
    end
end

for r=1:size(Patch_indy,1)
    for c=1:size(Patch_indx,1)
        im_patch = load(strcat('img_f',sprintf('_y%i_x%i',Patch_indy(r),Patch_indx(c))));
        im_patch = cell2mat(im_patch.new_img_global.f);

        im_patch_bordCut = im_patch(Cut_indy.start(r):Cut_indy.end(r),Cut_indx.start(c):Cut_indx.end(c));
        complete_img(Write_indy.start(r):Write_indy.end(r),Write_indx.start(c):Write_indx.end(c)) = im_patch_bordCut;
        fprintf('write_y%i_to%i_x%i_to_%i\n',Write_indy.start(r),Write_indy.end(r),Write_indx.start(c),Write_indx.end(c));
    end
end
t1 = toc;

cd(cur_dir);
compl_class = Assign_Color_to_Class(complete_img);
graph_pred_global_g = compl_class;
figure; imshow(graph_pred_global_g);

name_img = strcat(cur_dir,'/res/48/complete_img.png');           
imwrite(graph_pred_global_g,name_img);

