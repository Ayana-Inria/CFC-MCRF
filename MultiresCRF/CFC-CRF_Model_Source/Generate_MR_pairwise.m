function [ PAIRWISE ] = Generate_MR_pairwise( x_f, x_c )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


% Initialize variables
size_x_f = [x_f.xsx, x_f.xsy]; % Size of the full image (x_f)
size_x_c = [x_c.xsx, x_c.xsy]; % Size of the small image (x_c)
patch_size = ceil(size_x_f(1) / size_x_c(1)); % Size of the patch

% Total number of pixels in x_f and x_c
num_pixels_f = size_x_f(1) * size_x_f(2);
num_pixels_c = size_x_c(1) * size_x_c(2);

% Initialize the row indices for the sparse matrix (where 1's will go)
row_indices = zeros(patch_size^2 * num_pixels_c, 1);
col_indices = zeros(patch_size^2 * num_pixels_c, 1);

% Loop through each pixel in x_c
count = 0;
for i = 1:size_x_c(1)
    for j = 1:size_x_c(2)
        % For each pixel in x_c, get the corresponding 6x6 patch in x_f
        row_start = (i-1)*patch_size + 1;
        col_start = (j-1)*patch_size + 1;
        
        % Indices for the corresponding 6x6 patch in x_f
        patch_rows = row_start:(row_start + patch_size - 1);
        patch_cols = col_start:(col_start + patch_size - 1);
        
        % Get the linear indices for this 6x6 patch in x_f
        [rows, cols] = meshgrid(patch_rows, patch_cols);
        linear_indices = sub2ind(size_x_f, rows(:), cols(:));
        
        % Set the row and column indices for the sparse matrix
        row_indices(count + 1 : count + patch_size^2) = linear_indices;
        col_indices(count + 1 : count + patch_size^2) = (j-1)*size_x_c(1) + i; % Column-major order, mapped to x_c pixel
        
        % Update count for the next column
        count = count + patch_size^2;
    end
end

% Create the sparse matrix
PAIRWISE = sparse(row_indices, col_indices, 1, num_pixels_f, num_pixels_c);

% The sparse matrix is now ready


